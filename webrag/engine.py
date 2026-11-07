"""The orchestrator: ingest / refresh pages, track changes, retrieve, and answer."""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

from .chunk import chunk_page
from .config import Settings
from .documents import DocumentError, mime_type, parse_document, source_url
from .extract import Page, extract
from .fetch import Fetcher, FetchResult, same_site_links
from .providers import LLM, Embedder, make_embedder, make_llm
from .store import Hit, Store

NOT_FOUND = "I couldn't find that in the indexed sources."

ANSWER_SYSTEM = f"""You are WebRAG, a research assistant that answers questions strictly from \
numbered excerpts of web pages and documents that the user has chosen to index.

Rules:
- Use ONLY the information in the <sources> block of the latest message. Never rely on outside knowledge, \
even if you are confident it is correct — the sources may intentionally differ from what you know.
- Cite the excerpt(s) supporting each claim with bracketed numbers like [1] or [2][3], placed right after the claim.
- If the excerpts do not contain the answer, reply with exactly: "{NOT_FOUND}" and nothing else.
- If excerpts disagree with each other, point out the disagreement and cite both.
- Be direct and concise. Use short markdown lists or tables only when they genuinely help."""

REWRITE_SYSTEM = """Rewrite the user's latest question into a single standalone search query, \
resolving pronouns and references using the conversation. Output only the query text."""

SUMMARY_SYSTEM = """You describe changes between two versions of a web page or document for a change log. \
Given a unified diff of the text, write one or two plain sentences stating concretely what changed \
(e.g. "The admission fee rose from Rs. 50,000 to Rs. 75,000."). Ignore trivial whitespace or formatting \
changes. Output only the sentences."""

STOPWORDS = set(
    "a an and are as at be by can could did do does for from had has have how i if in into is it its "
    "me my of on or our so than that the their them then there these they this to was we were what when "
    "where which who whom why will with would you your about tell please according text page".split()
)


@dataclass
class ChangeEvent:
    url: str
    status: str  # new | changed | unchanged | error
    source_id: int | None = None
    title: str = ""
    via_304: bool = False
    diff: list[str] = field(default_factory=list)
    added: int = 0
    removed: int = 0
    summary: str = ""
    chunks_added: int = 0
    chunks_removed: int = 0
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def text_diff(old: str, new: str, max_lines: int = 300) -> tuple[list[str], int, int]:
    lines = list(difflib.unified_diff(old.splitlines(), new.splitlines(), "before", "after", n=1, lineterm=""))
    added = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more diff lines)"]
    return lines, added, removed


def fts_query(text: str) -> str:
    terms = [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS and len(t) > 1]
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))


class Engine:
    def __init__(self, settings: Settings | None = None, store: Store | None = None,
                 llm: LLM | None | bool = True, embedder: Embedder | None | bool = True,
                 transport=None):
        self.s = settings or Settings.from_env()
        self.store = store or Store(self.s.db_path)
        self.llm = make_llm(self.s) if llm is True else (llm or None)
        self.embedder = make_embedder(self.s) if embedder is True else (embedder or None)
        self._transport = transport  # for tests (httpx.MockTransport)
        self._ingest_lock = asyncio.Lock()
        self._check_embedding_model()

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        return {
            "llm": self.llm.name if self.llm else None,
            "embedder": self.embedder.name if self.embedder else None,
            "retrieval": "hybrid (BM25 + vectors)" if self.embedder else "keyword (BM25)",
            "sources": len(self.store.sources()),
        }

    def _check_embedding_model(self) -> None:
        current = self.embedder.name if self.embedder else ""
        if self.store.get_meta("embed_model") != current:
            self.store.clear_embeddings()
            self.store.set_meta("embed_model", current)

    def _fetcher(self) -> Fetcher:
        return Fetcher(self.s.user_agent, self.s.fetch_timeout, transport=self._transport)

    # ------------------------------------------------------------------ ingest
    async def add(self, urls: list[str], depth: int = 0, summarize: bool = True) -> list[ChangeEvent]:
        """Index new URLs (optionally crawling same-site links `depth` levels deep)."""
        events: list[ChangeEvent] = []
        seen: set[str] = set()
        frontier = [u.strip() for u in urls if u.strip()]
        async with self._ingest_lock, self._fetcher() as f:
            for level in range(depth + 1):
                batch = [u for u in dict.fromkeys(frontier) if u not in seen][: self.s.max_pages - len(seen)]
                if not batch:
                    break
                seen.update(batch)
                if level > 0:  # only crawled links are subject to robots.txt; explicit URLs are the user's call
                    allowed = await asyncio.gather(*(f.allowed(u) for u in batch))
                    batch = [u for u, ok in zip(batch, allowed) if ok]
                results = await asyncio.gather(*(self._fetch_known(f, u) for u in batch))
                frontier = []
                for res in results:
                    ev, links = await self._process(res, summarize)
                    events.append(ev)
                    if level < depth:
                        frontier.extend(same_site_links(links, res.requested))
            await asyncio.to_thread(self._embed_pending)
        return events

    async def add_document(self, filename: str, data: bytes, source_id: int | None = None,
                           summarize: bool = True) -> ChangeEvent:
        """Index an uploaded document. Uploading a file with the same name again (or passing the
        `source_id` it replaces) records a new version, with a diff and change summary."""
        filename = Path(filename.replace("\\", "/")).name.strip() or "document"
        row = self.store.source(source_id) if source_id is not None else None
        if source_id is not None and (row is None or row["kind"] != "file"):
            return ChangeEvent(source_url(filename), "error", error="That source is not an uploaded document.")
        url = row["url"] if row else source_url(filename)
        if len(data) > self.s.max_upload_mb * 1024 * 1024:
            return ChangeEvent(url, "error", source_id, filename,
                               error=f"File is larger than the {self.s.max_upload_mb:g} MB limit.")
        try:
            page = await asyncio.to_thread(parse_document, filename, data)
        except DocumentError as e:
            return ChangeEvent(url, "error", source_id, filename, error=str(e))

        async with self._ingest_lock:
            sid = self.store.upsert_source(url, kind="file")
            fields = dict(title=filename, status="ok", error=None, checked_at=time.time())
            ev = await self._ingest_page(sid, page, fields, summarize)
            # Keep the original bytes so citations can open the file (and jump to the cited PDF page).
            self.store.save_file(sid, filename, mime_type(filename), data)
            await asyncio.to_thread(self._embed_pending)
        return ev

    async def refresh(self, source_ids: list[int] | None = None, summarize: bool = True) -> list[ChangeEvent]:
        """Re-check indexed web pages; only changed pages are re-chunked / re-embedded.
        Uploaded documents have nothing to re-fetch: they change when a new version is uploaded."""
        rows = [r for r in self.store.sources()
                if (r["kind"] or "web") == "web" and (source_ids is None or r["id"] in source_ids)]
        async with self._ingest_lock, self._fetcher() as f:
            results = await asyncio.gather(*(self._fetch_known(f, r["url"]) for r in rows))
            events = [(await self._process(res, summarize))[0] for res in results]
            await asyncio.to_thread(self._embed_pending)
        return events

    async def _fetch_known(self, f: Fetcher, url: str) -> FetchResult:
        row = self.store.source_by_url(url)
        # Only use conditional GET when we hold usable content; an empty/broken entry must be re-read in full.
        if row and row["content_hash"] and self.store.chunk_hashes(row["id"]):
            return await f.fetch(url, row["etag"], row["last_modified"])
        return await f.fetch(url)

    async def _process(self, res: FetchResult, summarize: bool) -> tuple[ChangeEvent, list[str]]:
        url = res.requested
        sid = self.store.upsert_source(url)
        now = time.time()
        row = self.store.source(sid)

        if res.error:
            self.store.update_source(sid, status="error", error=res.error, checked_at=now)
            return ChangeEvent(url, "error", sid, row["title"] or "", error=res.error), []

        if res.not_modified:
            self.store.update_source(sid, status="ok", error=None, checked_at=now)
            return ChangeEvent(url, "unchanged", sid, row["title"] or "", via_304=True), []

        page = extract(res.html, res.url)
        text = page.text
        if not text.strip():
            err = "No readable text found on the page (it may require JavaScript to render)."
            self.store.update_source(sid, status="error", error=err, checked_at=now)
            return ChangeEvent(url, "error", sid, page.title, error=err), page.links
        fields = dict(title=page.title, etag=res.etag, last_modified=res.last_modified,
                      status="ok", error=None, checked_at=now)
        return await self._ingest_page(sid, page, fields, summarize), page.links

    async def _ingest_page(self, sid: int, page: Page, fields: dict, summarize: bool) -> ChangeEvent:
        """Store freshly extracted content (web page or document): version, diff, summary, chunk sync."""
        row = self.store.source(sid)
        url, text, now = row["url"], page.text, time.time()
        h = hashlib.sha256(text.encode()).hexdigest()

        if row["content_hash"] == h:
            self.store.update_source(sid, **fields)
            return ChangeEvent(url, "unchanged", sid, page.title)

        prev = self.store.latest_version(sid)
        vid = self.store.add_version(sid, h, text)
        chunks = chunk_page(page, self.s.chunk_chars)
        before = self.store.chunk_hashes(sid)
        inserted = self.store.sync_chunks(sid, [(c.hash, c.section, c.text) for c in chunks])
        removed = len(set(before) - {c.hash for c in chunks})
        self.store.update_source(sid, content_hash=h, changed_at=now, **fields)

        if prev is None:
            return ChangeEvent(url, "new", sid, page.title, chunks_added=len(inserted))

        diff, n_add, n_del = text_diff(prev["text"], text)
        ev = ChangeEvent(url, "changed", sid, page.title, diff=diff, added=n_add, removed=n_del,
                         chunks_added=len(inserted), chunks_removed=removed)
        if summarize and self.llm and diff:
            try:
                ev.summary = await asyncio.to_thread(
                    self.llm.complete, SUMMARY_SYSTEM, "\n".join(diff[:200])
                )
                self.store.set_version_summary(vid, ev.summary)
            except Exception as e:  # a summary is a nice-to-have; never fail the refresh for it
                ev.summary = ""
                print(f"[webrag] change summary failed: {e}")
        return ev

    def _embed_pending(self) -> None:
        if not self.embedder:
            return
        rows = self.store.chunks_missing_embeddings()
        if not rows:
            return
        texts = [f"{r['title'] or ''} > {r['section'] or ''}\n{r['text']}" for r in rows]
        try:
            vecs = self.embedder.embed(texts)
        except Exception as e:
            # Keyword search still works; missing vectors are retried on the next add/refresh.
            print(f"[webrag] embedding failed ({type(e).__name__}: {e}); keyword search only for now")
            return
        self.store.set_embeddings([(r["id"], v) for r, v in zip(rows, vecs)])

    # ------------------------------------------------------------------ history
    def history(self, source_id: int) -> list[dict]:
        versions = self.store.versions(source_id)
        out = []
        for i, v in enumerate(versions):
            older = versions[i + 1] if i + 1 < len(versions) else None
            diff, a, d = text_diff(older["text"], v["text"]) if older else ([], 0, 0)
            out.append({
                "id": v["id"], "fetched_at": v["fetched_at"], "summary": v["summary"] or "",
                "diff": diff, "added": a, "removed": d, "initial": older is None,
            })
        return out

    def remove(self, source_id: int) -> None:
        self.store.delete_source(source_id)

    # ------------------------------------------------------------------ retrieval
    def search(self, query: str, k: int | None = None) -> list[Hit]:
        k = k or self.s.top_k
        pool = max(k * 4, 20)
        ranked: list[list[tuple[int, float]]] = [self.store.keyword_search(fts_query(query), pool)]
        if self.embedder:
            try:
                qv = self.embedder.embed([query], kind="query")[0]
                ranked.append(self.store.vector_search(qv, pool))
            except Exception as e:
                print(f"[webrag] vector search unavailable, using keywords only: {e}")

        # Reciprocal Rank Fusion: robust way to merge rankings with incomparable scores.
        fused: dict[int, float] = {}
        for ranking in ranked:
            for rank, (cid, _) in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        hits = self.store.hits(top)
        result = []
        for cid in top:
            if cid in hits:
                hits[cid].score = fused[cid]
                result.append(hits[cid])
        return result

    # ------------------------------------------------------------------ answering
    def ask(self, question: str, history: list[dict] | None = None) -> Iterator[dict]:
        """Yields events: {"type": "sources"}, {"type": "token"}..., {"type": "done"}."""
        history = [m for m in (history or []) if m.get("role") in ("user", "assistant") and m.get("content")][-6:]

        query = question
        if history and self.llm:
            try:
                convo = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history)
                query = self.llm.complete(REWRITE_SYSTEM, f"{convo}\nuser: {question}") or question
            except Exception:
                query = question

        hits = self.search(query)
        yield {"type": "sources", "query": query, "sources": [self._hit_dict(i + 1, h) for i, h in enumerate(hits)]}

        if not hits:
            yield {"type": "token", "text": NOT_FOUND}
            yield {"type": "done"}
            return

        if not self.llm:
            yield {"type": "token", "text": "_No language model configured - showing the most relevant passages._\n\n"
                   + "\n\n".join(f"**[{i + 1}]** {h.text[:400]}" for i, h in enumerate(hits[:3]))}
            yield {"type": "done"}
            return

        context = "\n\n".join(
            f"[{i + 1}] {h.title}{' — ' + h.section if h.section else ''}\n"
            + (f"Uploaded document, last updated: {_fmt_time(h.changed_at)}\n" if h.kind == "file"
               else f"URL: {h.url}\nLast checked: {_fmt_time(h.checked_at)}\n")
            + h.text
            for i, h in enumerate(hits)
        )
        messages = [*history, {"role": "user", "content": f"<sources>\n{context}\n</sources>\n\nQuestion: {question}"}]
        try:
            for piece in self.llm.stream(ANSWER_SYSTEM, messages):
                yield {"type": "token", "text": piece}
        except Exception as e:
            yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
        yield {"type": "done"}

    def answer(self, question: str, history: list[dict] | None = None) -> tuple[str, list[dict]]:
        text, sources = [], []
        for ev in self.ask(question, history):
            if ev["type"] == "sources":
                sources = ev["sources"]
            elif ev["type"] == "token":
                text.append(ev["text"])
            elif ev["type"] == "error":
                raise RuntimeError(ev["message"])
        return "".join(text), sources

    @staticmethod
    def _hit_dict(n: int, h: Hit) -> dict:
        return {"n": n, "url": h.url, "title": h.title, "section": h.section, "text": h.text,
                "checked_at": h.checked_at, "changed_at": h.changed_at, "score": round(h.score, 4),
                "kind": h.kind, "source_id": h.source_id}


def _fmt_time(ts: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(ts)) if ts else "unknown"
