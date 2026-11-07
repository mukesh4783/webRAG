"""Uploaded documents: parsing for each format, versioning, citations and the upload API.
Test files are generated in memory, so no fixtures need to be checked in."""

from __future__ import annotations

import asyncio
import io

import pytest

from webrag.config import Settings
from webrag.documents import DocumentError, parse_document
from webrag.engine import Engine
from tests.test_webrag import FakeEmbedder, FakeLLM


# ------------------------------------------------------------------ file builders
def make_pdf(pages: list[str]) -> bytes:
    """A minimal valid PDF with one line of Helvetica text per page."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream.decode()}\nendstream")
        content_id = len(objs)
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def make_docx(fee: str = "50,000") -> bytes:
    import docx

    d = docx.Document()
    d.add_heading("Mock University Handbook", 0)
    d.add_heading("Admissions", 1)
    d.add_paragraph(f"The admission fee for 2026 is Rs. {fee}.")
    d.add_heading("Hostels", 2)
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Hostel", "Price"
    t.cell(1, 0).text, t.cell(1, 1).text = "North", "Rs. 9,000"
    d.add_heading("Campus", 1)
    d.add_paragraph("The library is open 24 hours.")
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def make_pptx() -> bytes:
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    s1 = prs.slides.add_slide(prs.slide_layouts[1])
    s1.shapes.title.text = "Quarterly Results"
    s1.placeholders[1].text = "Revenue grew 12% to $4.2M"
    s1.notes_slide.notes_text_frame.text = "Mention the new APAC office."
    s2 = prs.slides.add_slide(prs.slide_layouts[5])
    s2.shapes.title.text = "Pricing"
    tbl = s2.shapes.add_table(2, 2, Inches(1), Inches(2), Inches(4), Inches(1)).table
    tbl.cell(0, 0).text, tbl.cell(0, 1).text = "Plan", "Price"
    tbl.cell(1, 0).text, tbl.cell(1, 1).text = "Pro", "$49/month"
    s3 = prs.slides.add_slide(prs.slide_layouts[5])
    s3.shapes.title.text = "Thank you"
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


@pytest.fixture
def eng(tmp_path):
    e = Engine(Settings(db_path=str(tmp_path / "d.db")), llm=FakeLLM(), embedder=FakeEmbedder())
    yield e
    e.store.close()


# ------------------------------------------------------------------ parsing
def test_pdf_sections_are_pages():
    page = parse_document("report.pdf", make_pdf(["Intro text here.", "The admission fee is Rs. 50,000."]))
    assert page.title == "report.pdf" and page.url == "upload://report.pdf"
    fee = next(b for b in page.blocks if "admission fee" in b.text)
    assert fee.section == "Page 2"


def test_docx_keeps_heading_hierarchy_and_tables():
    page = parse_document("handbook.docx", make_docx())
    fee = next(b for b in page.blocks if "admission fee" in b.text)
    assert fee.section == "Mock University Handbook > Admissions"
    row = next(b for b in page.blocks if "North" in b.text)
    assert row.text == "North | Rs. 9,000" and row.section.endswith("Admissions > Hostels")
    lib = next(b for b in page.blocks if "library" in b.text)
    assert lib.section == "Mock University Handbook > Campus"


def test_pptx_slides_tables_and_notes():
    page = parse_document("deck.pptx", make_pptx())
    text = page.text
    assert "Revenue grew 12%" in text and "Speaker notes: Mention the new APAC office." in text
    assert "Pro | $49/month" in text
    assert next(b for b in page.blocks if "Revenue" in b.text).section == "Slide 1: Quarterly Results"
    # A title-only slide still yields searchable content.
    assert any(b.text == "Thank you" and not b.is_heading for b in page.blocks)


def test_markdown_headings():
    page = parse_document("notes.md", b"# Policies\n\n## Refunds\nRefunds take\n5 days.\n\n- Item one\n")
    refund = next(b for b in page.blocks if "Refunds take" in b.text)
    assert refund.text == "Refunds take 5 days." and refund.section == "Policies > Refunds"


def test_bad_files_raise_friendly_errors():
    with pytest.raises(DocumentError, match="Unsupported"):
        parse_document("x.exe", b"MZ")
    with pytest.raises(DocumentError, match="Could not read"):
        parse_document("broken.docx", b"not a zip")
    with pytest.raises(DocumentError, match="no text"):
        parse_document("empty.txt", b"   \n\n")


# ------------------------------------------------------------------ engine
def test_upload_answer_and_new_version(eng):
    ev = asyncio.run(eng.add_document("handbook.docx", make_docx("50,000")))
    assert ev.status == "new" and ev.chunks_added > 0

    answer, sources = eng.answer("What is the admission fee?")
    assert "Rs. 50,000" in answer
    assert sources[0]["kind"] == "file" and sources[0]["source_id"] == ev.source_id
    assert "Admissions" in sources[0]["section"]

    # Same file again: nothing changes.
    again = asyncio.run(eng.add_document("handbook.docx", make_docx("50,000")))
    assert again.status == "unchanged" and again.source_id == ev.source_id

    # Updated file with the same name: a new version with a diff and summary.
    upd = asyncio.run(eng.add_document("handbook.docx", make_docx("75,000")))
    assert upd.status == "changed" and upd.source_id == ev.source_id
    assert any("75,000" in l and l.startswith("+") for l in upd.diff)
    assert upd.summary == "The fee changed."
    answer, _ = eng.answer("What is the admission fee?")
    assert "Rs. 75,000" in answer
    assert len(eng.history(ev.source_id)) == 2


def test_replace_by_id_keeps_identity(eng):
    ev = asyncio.run(eng.add_document("handbook.docx", make_docx("50,000")))
    upd = asyncio.run(eng.add_document("handbook-v2.docx", make_docx("75,000"), source_id=ev.source_id))
    assert upd.status == "changed" and upd.source_id == ev.source_id
    row = eng.store.source(ev.source_id)
    assert row["title"] == "handbook-v2.docx" and eng.store.file(ev.source_id)["filename"] == "handbook-v2.docx"
    assert len(eng.store.sources()) == 1


def test_refresh_skips_uploaded_documents(eng):
    asyncio.run(eng.add_document("deck.pptx", make_pptx()))
    assert asyncio.run(eng.refresh()) == []


def test_unreadable_upload_is_an_error_event_and_not_stored(eng):
    ev = asyncio.run(eng.add_document("scan.pdf", b"%PDF-1.4 garbage"))
    assert ev.status == "error" and ev.error
    assert eng.store.sources() == []


# ------------------------------------------------------------------ API
def test_upload_api_and_file_download(eng):
    from fastapi.testclient import TestClient

    from webrag.server import create_app

    client = TestClient(create_app(eng))
    pdf = make_pdf(["The admission fee is Rs. 50,000."])
    r = client.post("/api/documents", params={"filename": "fees.pdf"}, content=pdf)
    assert r.status_code == 200 and r.json()[0]["status"] == "new"
    sid = r.json()[0]["source_id"]

    [src] = client.get("/api/sources").json()
    assert src["kind"] == "file" and src["file_size"] == len(pdf) and src["n_chunks"] > 0

    f = client.get(f"/api/sources/{sid}/file")
    assert f.status_code == 200 and f.content == pdf and f.headers["content-type"] == "application/pdf"

    assert client.post("/api/documents", params={"filename": "x.exe"}, content=b"MZ").status_code == 400
    assert client.post("/api/documents", params={"filename": "a.pdf"}, content=b"").status_code == 400
    assert client.post("/api/documents", params={"filename": "a.pdf", "replace": 999}, content=pdf).status_code == 404

    eng.s.max_upload_mb = 0.0001
    assert client.post("/api/documents", params={"filename": "big.pdf"}, content=pdf).status_code == 413

    assert client.delete(f"/api/sources/{sid}").status_code == 200
    assert client.get(f"/api/sources/{sid}/file").status_code == 404
