"""Turn uploaded documents (PDF, Word, PowerPoint, plain text) into the same structure-aware
`Page` the HTML extractor produces, so chunking, change tracking and citations work unchanged.

Each format maps its natural structure onto heading paths:

* PDF        -> one section per page        ("Page 3")
* PowerPoint -> one section per slide       ("Slide 4: Pricing"), speaker notes included
* Word       -> real heading hierarchy      ("Policies > Refunds"), tables kept as rows
* .txt / .md -> markdown headings, if any

Legacy binary formats (.doc, .ppt) are converted with LibreOffice when it is installed.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .extract import Block, Page

SUPPORTED = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt": "application/vnd.ms-powerpoint",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
}
LEGACY = {".doc": "docx", ".ppt": "pptx"}
WS = re.compile(r"[ \t\r\f\v ]+")


class DocumentError(ValueError):
    """The file could not be read; the message is safe to show to the user."""


def extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def is_supported(filename: str) -> bool:
    return extension(filename) in SUPPORTED


def mime_type(filename: str) -> str:
    return SUPPORTED.get(extension(filename), "application/octet-stream")


def source_url(filename: str) -> str:
    """Stable key for an uploaded file: re-uploading the same name records a new version."""
    return f"upload://{filename}"


def _clean(text: str) -> str:
    return WS.sub(" ", text).strip()


def parse_document(filename: str, data: bytes) -> Page:
    ext = extension(filename)
    if ext not in SUPPORTED:
        raise DocumentError(f"Unsupported file type {ext or '(none)'}. Supported: {', '.join(sorted(SUPPORTED))}")
    if not data:
        raise DocumentError("The file is empty.")
    if ext in LEGACY:
        data, ext = _convert_legacy(data, ext), "." + LEGACY[ext]
    parser = {".pdf": _pdf, ".docx": _docx, ".pptx": _pptx, ".txt": _text, ".md": _text}[ext]
    try:
        blocks = parser(data)
    except DocumentError:
        raise
    except Exception as e:  # corrupt / password-protected / mislabelled files
        raise DocumentError(f"Could not read {filename}: {type(e).__name__}: {e}") from e
    return Page(url=source_url(filename), title=filename, blocks=blocks)


# ------------------------------------------------------------------ PDF
def _pdf(data: bytes) -> list[Block]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise DocumentError("This PDF is password-protected.") from None
    blocks: list[Block] = []
    for n, page in enumerate(reader.pages, 1):
        for para in _paragraphs(page.extract_text() or ""):
            blocks.append(Block(para, f"Page {n}"))
    if not blocks:
        raise DocumentError("No extractable text in this PDF (it may be a scan; OCR is not supported).")
    return blocks


def _paragraphs(text: str) -> list[str]:
    """PDF text arrives as hard-wrapped lines. Re-join lines into paragraphs, splitting on blank
    lines and on lines that clearly end a sentence before a short/indented next line."""
    out, buf = [], []
    for raw in text.splitlines():
        line = _clean(raw)
        if not line:
            if buf:
                out.append(" ".join(buf))
                buf = []
            continue
        if buf and buf[-1].endswith("-") and not buf[-1].endswith(" -"):
            buf[-1] = buf[-1][:-1] + line  # de-hyphenate words broken across lines
        else:
            buf.append(line)
        if line.endswith((".", "?", "!", ":")) and len(line) < 60:
            out.append(" ".join(buf))
            buf = []
    if buf:
        out.append(" ".join(buf))
    return [p for p in out if len(p) > 1]


# ------------------------------------------------------------------ Word
def _docx(data: bytes) -> list[Block]:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = docx.Document(io.BytesIO(data))
    path: list[tuple[int, str]] = []
    blocks: list[Block] = []

    def section() -> str:
        return " > ".join(h for _, h in path)

    for el in doc.element.body.iterchildren():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(el, doc)
            text = _clean(para.text)
            if not text:
                continue
            level = _heading_level(para.style.name if para.style is not None else "")
            if level:
                while path and path[-1][0] >= level:
                    path.pop()
                path.append((level, text))
                blocks.append(Block(text, section(), is_heading=True))
            else:
                blocks.append(Block(text, section()))
        elif tag == "tbl":
            for row in Table(el, doc).rows:
                cells = []
                for cell in row.cells:  # merged cells repeat; keep each value once per row
                    t = _clean(cell.text)
                    if t and (not cells or cells[-1] != t):
                        cells.append(t)
                if cells:
                    blocks.append(Block(" | ".join(cells), section()))
    if not any(not b.is_heading for b in blocks):
        raise DocumentError("No readable text found in this Word document.")
    return blocks


def _heading_level(style: str) -> int:
    """0 = body text. "Title" ranks above "Heading 1", so levels are shifted by one."""
    s = style.lower()
    if s == "title":
        return 1
    m = re.match(r"heading\s*(\d)", s)
    return int(m.group(1)) + 1 if m else 0


# ------------------------------------------------------------------ PowerPoint
def _pptx(data: bytes) -> list[Block]:
    from pptx import Presentation

    prs = Presentation(io.BytesIO(data))
    blocks: list[Block] = []
    for n, slide in enumerate(prs.slides, 1):
        title_shape = slide.shapes.title
        title = _clean(title_shape.text_frame.text) if title_shape is not None and title_shape.has_text_frame else ""
        sec = f"Slide {n}: {title}" if title else f"Slide {n}"
        blocks.append(Block(sec, sec, is_heading=True))
        start = len(blocks)
        for shape in _walk(slide.shapes):
            if title_shape is not None and shape.shape_id == title_shape.shape_id:
                continue
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    cells = [_clean(c.text) for c in row.cells]
                    if any(cells):
                        blocks.append(Block(" | ".join(c for c in cells if c), sec))
            elif getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                for p in shape.text_frame.paragraphs:
                    text = _clean("".join(r.text for r in p.runs) or "")
                    if text:
                        blocks.append(Block(text, sec))
        if slide.has_notes_slide:
            notes = _clean(slide.notes_slide.notes_text_frame.text or "")
            if notes:
                blocks.append(Block(f"Speaker notes: {notes}", sec))
        if len(blocks) == start and title:  # title-only slide: keep the title searchable as content
            blocks.append(Block(title, sec))
    if not any(not b.is_heading for b in blocks):
        raise DocumentError("No readable text found in this presentation.")
    return blocks


def _walk(shapes):
    for shape in shapes:
        if shape.shape_type == 6:  # MSO_SHAPE_TYPE.GROUP
            yield from _walk(shape.shapes)
        else:
            yield shape


# ------------------------------------------------------------------ plain text / markdown
def _text(data: bytes) -> list[Block]:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    path: list[tuple[int, str]] = []
    blocks: list[Block] = []
    para: list[str] = []

    def flush() -> None:
        if para:
            blocks.append(Block(" ".join(para), " > ".join(h for _, h in path)))
            para.clear()

    for raw in text.splitlines():
        line = _clean(raw)
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, title))
            blocks.append(Block(title, " > ".join(h for _, h in path), is_heading=True))
        elif not line:
            flush()
        elif re.match(r"^([-*•]|\d+[.)])\s+", line):
            flush()
            para.append(line)
            flush()
        else:
            para.append(line)
    flush()
    if not any(not b.is_heading for b in blocks):
        raise DocumentError("The file contains no text.")
    return blocks


# ------------------------------------------------------------------ legacy .doc / .ppt
def _soffice() -> str | None:
    found = shutil.which("soffice") or shutil.which("libreoffice")
    if found:
        return found
    for base in (os.environ.get("ProgramFiles", ""), os.environ.get("ProgramFiles(x86)", ""), "/Applications"):
        for rel in ("LibreOffice/program/soffice.exe", "LibreOffice.app/Contents/MacOS/soffice"):
            p = Path(base) / rel
            if base and p.is_file():
                return str(p)
    return None


def _convert_legacy(data: bytes, ext: str) -> bytes:
    exe = _soffice()
    target = LEGACY[ext]
    if not exe:
        raise DocumentError(
            f"Old-style {ext} files need LibreOffice to convert them. Install LibreOffice, "
            f"or re-save the file as .{target} and upload that."
        )
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"input{ext}"
        src.write_bytes(data)
        try:
            subprocess.run(
                [exe, "--headless", "--norestore", "--convert-to", target, "--outdir", tmp, str(src)],
                check=True, capture_output=True, timeout=120,
            )
        except (subprocess.SubprocessError, OSError) as e:
            raise DocumentError(f"LibreOffice could not convert this {ext} file ({type(e).__name__}).") from e
        out = Path(tmp) / f"input.{target}"
        if not out.is_file():
            raise DocumentError(f"LibreOffice could not convert this {ext} file.")
        return out.read_bytes()
