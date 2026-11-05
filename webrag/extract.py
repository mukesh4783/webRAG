"""Turn raw HTML into clean, structure-aware text.

Instead of dumping every string on the page, we keep only the readable content
blocks (headings, paragraphs, list items, table rows, code) and remember which
heading each block lives under. That gives us:

* cleaner diffs (one line per block, no menu/footer noise),
* chunks that never straddle unrelated sections,
* citations that can say *where* on the page an answer came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urldefrag, urlparse

from bs4 import BeautifulSoup, Tag

NOISE_TAGS = [
    "script", "style", "noscript", "template", "svg", "canvas", "iframe",
    "nav", "footer", "header", "aside", "form", "button", "select",
]
NOISE_HINTS = re.compile(
    r"(^|[\s_-])(nav|navbar|menu|footer|sidebar|cookie|banner|breadcrumb|"
    r"advert|promo|share|social|skip|toc|mw-editsection|reference)s?($|[\s_-])",
    re.I,
)
PROTECTED = {"html", "body", "main", "article"}
HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
BLOCKS = HEADINGS | {"p", "li", "pre", "blockquote", "tr", "dt", "dd", "figcaption", "caption"}
WS = re.compile(r"\s+")


@dataclass
class Block:
    text: str
    section: str  # heading path, e.g. "Admissions > Fees"
    is_heading: bool = False


@dataclass
class Page:
    url: str
    title: str
    blocks: list[Block] = field(default_factory=list)
    links: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Canonical text used for hashing and diffing: one block per line."""
        lines = []
        for b in self.blocks:
            lines.append(("#" * 2 + " " + b.text) if b.is_heading else b.text)
        return "\n".join(lines)


def _clean(text: str) -> str:
    return WS.sub(" ", text).strip()


def _is_noise(tag: Tag) -> bool:
    if tag.attrs is None:
        return False
    if tag.get("aria-hidden") == "true" or tag.get("hidden") is not None:
        return True
    role = (tag.get("role") or "").lower()
    if role in {"navigation", "banner", "contentinfo", "complementary", "search"}:
        return True
    hint = " ".join(tag.get("class") or []) + " " + (tag.get("id") or "")
    return bool(hint.strip()) and bool(NOISE_HINTS.search(hint))


def _content_root(soup: BeautifulSoup) -> Tag:
    for selector in ("main", "article", "[role=main]", "#content", "#main", ".content"):
        found = soup.select_one(selector)
        if found and len(found.get_text(strip=True)) > 200:
            return found
    return soup.body or soup


def extract(html: str, url: str) -> Page:
    soup = BeautifulSoup(html, "lxml")

    title = _clean(soup.title.get_text()) if soup.title else ""
    links = _links(soup, url)

    for tag in soup.find_all(NOISE_TAGS):
        tag.decompose()

    # Pick the content root first, then strip class/id-hinted chrome *inside* it. Class hints are
    # fuzzy (Wikipedia's <html> has "...-main-menu-disabled"), so never drop structural containers
    # or anything holding a large share of the page's text.
    root = _content_root(soup)
    total = len(root.get_text(" ", strip=True)) or 1
    for tag in root.find_all(True):
        if tag.decomposed or tag.name in PROTECTED:
            continue
        if _is_noise(tag) and len(tag.get_text(" ", strip=True)) < 0.3 * total:
            tag.decompose()
    path: list[tuple[int, str]] = []  # stack of (level, heading text)
    blocks: list[Block] = []
    seen_text = 0

    for el in root.find_all(BLOCKS):
        # Skip blocks nested in other blocks (e.g. <p> inside <li>): the outer one covers them.
        if el.name not in HEADINGS and el.find_parent(BLOCKS - HEADINGS):
            continue
        if el.name == "tr":
            cells = [_clean(c.get_text(" ")) for c in el.find_all(["th", "td"])]
            text = " | ".join(c for c in cells if c)
        else:
            text = _clean(el.get_text(" "))
        if not text:
            continue

        if el.name in HEADINGS:
            level = int(el.name[1])
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, text))
            blocks.append(Block(text, " > ".join(h for _, h in path), is_heading=True))
        else:
            blocks.append(Block(text, " > ".join(h for _, h in path)))
        seen_text += len(text)

    # Fallback for pages that put their text in bare <div>s.
    total = len(_clean(root.get_text(" ")))
    if total and seen_text < 0.4 * total:
        blocks = [
            Block(line, "")
            for line in (_clean(s) for s in root.get_text("\n").split("\n"))
            if len(line) > 1
        ]

    if not title and blocks:
        title = blocks[0].text[:80]
    return Page(url=url, title=title or url, blocks=blocks, links=links)


def _links(soup: BeautifulSoup, base: str) -> list[str]:
    out, seen = [], set()
    for a in soup.find_all("a", href=True):
        href = urldefrag(urljoin(base, a["href"]))[0]
        if urlparse(href).scheme in {"http", "https"} and href not in seen:
            seen.add(href)
            out.append(href)
    return out
