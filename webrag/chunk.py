"""Section-aware chunking.

Chunks are built from consecutive blocks of the *same section*, so a chunk about
"Admission fees" never gets glued to the start of "Campus life". Each chunk keeps
its heading path, which is prepended when embedding and shown in citations.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .extract import Page


@dataclass(frozen=True)
class Chunk:
    section: str
    text: str

    @property
    def hash(self) -> str:
        return hashlib.sha256(f"{self.section}\n{self.text}".encode()).hexdigest()[:32]

    def for_embedding(self, title: str) -> str:
        head = " > ".join(p for p in (title, self.section) if p)
        return f"{head}\n{self.text}" if head else self.text


def _split_long(text: str, limit: int) -> list[str]:
    """Split an oversized block on sentence boundaries, falling back to hard cuts."""
    if len(text) <= limit:
        return [text]
    parts, buf = [], ""
    for sentence in text.replace("? ", "?\n").replace("! ", "!\n").replace(". ", ".\n").split("\n"):
        if buf and len(buf) + len(sentence) + 1 > limit:
            parts.append(buf)
            buf = ""
        buf = f"{buf} {sentence}".strip()
        while len(buf) > limit:
            parts.append(buf[:limit])
            buf = buf[limit:]
    if buf:
        parts.append(buf)
    return parts


def chunk_page(page: Page, max_chars: int = 1200) -> list[Chunk]:
    chunks: list[Chunk] = []
    section, buf = None, []

    def flush() -> None:
        if buf:
            chunks.append(Chunk(section or "", "\n".join(buf)))
            buf.clear()

    for block in page.blocks:
        if block.is_heading:
            flush()
            section = block.section
            continue
        if block.section != section:
            flush()
            section = block.section
        for piece in _split_long(block.text, max_chars):
            if buf and sum(len(b) + 1 for b in buf) + len(piece) > max_chars:
                flush()
            buf.append(piece)
    flush()

    # De-duplicate identical chunks (repeated boilerplate) while preserving order.
    seen, unique = set(), []
    for c in chunks:
        if c.hash not in seen:
            seen.add(c.hash)
            unique.append(c)
    return unique
