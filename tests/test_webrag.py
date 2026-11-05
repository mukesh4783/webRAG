"""Offline tests: HTTP is mocked, the LLM and embedder are deterministic fakes."""

from __future__ import annotations

import asyncio
import hashlib
import re

import httpx
import numpy as np
import pytest

from webrag.chunk import chunk_page
from webrag.config import Settings
from webrag.engine import NOT_FOUND, Engine, fts_query, text_diff
from webrag.extract import extract

PAGE = """<html><head><title>Mock University</title></head><body>
<nav><a href="/a">Home</a> <a href="/docs/fees.html">Fees</a></nav>
<main>
  <h1>Welcome to Mock University</h1>
  <p>Our university offers top-tier education.</p>
  <h2>Admission Details</h2>
  <p>The current admission fee for the 2026 academic year is Rs. {fee}.</p>
  <ul><li>Deadline: <b>March 31</b></li><li>Interview required</li></ul>
  <h2>Campus</h2>
  <p>The campus has a library open 24 hours.</p>
  <table><tr><th>Hostel</th><th>Price</th></tr><tr><td>North</td><td>Rs. 9,000</td></tr></table>
</main>
<footer>Copyright Mock University</footer>
</body></html>"""


class FakeEmbedder:
    name = "fake:bow"

    def embed(self, texts, kind="document"):
        out = []
        for t in texts:
            v = np.zeros(256, dtype=np.float32)
            for w in re.findall(r"\w+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
            out.append(v)
        return out


class FakeLLM:
    name = "fake:llm"

    def __init__(self):
        self.calls = []

    def complete(self, system, prompt):
        self.calls.append(("complete", prompt))
        return "The fee changed."

    def stream(self, system, messages):
        content = messages[-1]["content"]
        self.calls.append(("stream", content))
        m = re.search(r"admission fee[^\n]*?(Rs\. [\d,]+)", content)
        answer = f"The fee is {m.group(1)} [1]." if m else NOT_FOUND
        for word in answer.split(" "):
            yield word + " "


@pytest.fixture
def site():
    state = {"fee": "50,000", "hits": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["hits"] += 1
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.path == "/missing":
            return httpx.Response(404)
        body = PAGE.format(fee=state["fee"])
        etag = '"' + hashlib.md5(body.encode()).hexdigest() + '"'
        if request.headers.get("if-none-match") == etag:
            return httpx.Response(304, headers={"etag": etag})
        return httpx.Response(200, text=body, headers={"content-type": "text/html", "etag": etag})

    return state, httpx.MockTransport(handler)


@pytest.fixture
def engine(tmp_path, site):
    _, transport = site
    s = Settings(db_path=str(tmp_path / "t.db"))
    llm = FakeLLM()
    eng = Engine(s, llm=llm, embedder=FakeEmbedder(), transport=transport)
    eng.fake_llm = llm
    yield eng
    eng.store.close()


# ------------------------------------------------------------------ extraction & chunking
def test_extract_keeps_content_and_drops_chrome():
    page = extract(PAGE.format(fee="50,000"), "http://x/docs/index.html")
    assert page.title == "Mock University"
    text = page.text
    assert "admission fee" in text and "Deadline: March 31" in text
    assert "Copyright" not in text and "Home" not in text
    assert "Hostel | Price" in text
    fee_block = next(b for b in page.blocks if "admission fee" in b.text)
    assert fee_block.section == "Welcome to Mock University > Admission Details"
    assert "http://x/docs/fees.html" in page.links


def test_chunks_respect_sections():
    page = extract(PAGE.format(fee="50,000"), "http://x/")
    chunks = chunk_page(page, max_chars=1200)
    sections = [c.section for c in chunks]
    assert any(s.endswith("Admission Details") for s in sections)
    assert any(s.endswith("Campus") for s in sections)
    for c in chunks:
        assert not ("admission fee" in c.text and "library" in c.text)


def test_long_blocks_are_split():
    html = "<p>" + " ".join(f"This is sentence number {i}." for i in range(300)) + "</p>"
    chunks = chunk_page(extract(html, "http://x/"), max_chars=500)
    assert len(chunks) > 5 and all(len(c.text) <= 500 for c in chunks)


def test_fts_query_sanitizes():
    assert fts_query('What is the "admission" fee?!') == '"admission" OR "fee"'
    assert fts_query("the of and") == ""


def test_text_diff_counts():
    lines, a, d = text_diff("a\nb\nc", "a\nB\nc")
    assert (a, d) == (1, 1) and "-b" in lines and "+B" in lines


# ------------------------------------------------------------------ end-to-end
def test_index_ask_change_and_reask(engine, site):
    state, _ = site
    url = "http://uni.test/docs/index.html"

    [ev] = asyncio.run(engine.add([url]))
    assert ev.status == "new" and ev.chunks_added > 0

    answer, sources = engine.answer("What is the admission fee?")
    assert "Rs. 50,000" in answer and sources
    assert "Admission Details" in sources[0]["section"]

    # Same content -> server answers 304, nothing re-embedded.
    [ev] = asyncio.run(engine.refresh())
    assert ev.status == "unchanged" and ev.via_304

    # Page changes -> diff, summary, only the affected chunk is re-embedded.
    state["fee"] = "75,000"
    n_chunks = len(engine.store.chunk_hashes(ev.source_id))
    [ev] = asyncio.run(engine.refresh())
    assert ev.status == "changed"
    assert any("50,000" in l and l.startswith("-") for l in ev.diff)
    assert any("75,000" in l and l.startswith("+") for l in ev.diff)
    assert ev.summary == "The fee changed."
    assert ev.chunks_added == 1 and ev.chunks_removed == 1
    assert len(engine.store.chunk_hashes(ev.source_id)) == n_chunks

    answer, _ = engine.answer("What is the admission fee?")
    assert "Rs. 75,000" in answer

    hist = engine.history(ev.source_id)
    assert len(hist) == 2 and hist[0]["summary"] == "The fee changed." and hist[1]["initial"]


def test_off_topic_question_is_refused_without_llm_call(tmp_path, site):
    _, transport = site
    eng = Engine(Settings(db_path=str(tmp_path / "k.db")), llm=FakeLLM(), embedder=None, transport=transport)
    asyncio.run(eng.add(["http://uni.test/"]))
    answer, sources = eng.answer("Who is Hugh Jackman?")
    assert answer == NOT_FOUND and sources == []
    assert not [c for c in eng.llm.calls if c[0] == "stream"]


def test_errors_are_reported(engine):
    [ev] = asyncio.run(engine.add(["http://uni.test/missing"]))
    assert ev.status == "error" and "404" in ev.error


def test_crawl_follows_same_site_links(engine):
    events = asyncio.run(engine.add(["http://uni.test/docs/index.html"], depth=1))
    urls = {e.url for e in events}
    assert "http://uni.test/docs/fees.html" in urls      # same section of the site
    assert "http://uni.test/a" not in urls                # outside the /docs/ prefix


def test_followup_question_is_rewritten(engine):
    asyncio.run(engine.add(["http://uni.test/"]))
    events = list(engine.ask("and the deadline?", [{"role": "user", "content": "What is the fee?"},
                                                   {"role": "assistant", "content": "Rs. 50,000 [1]"}]))
    assert events[0]["type"] == "sources" and events[0]["query"] == "The fee changed."  # FakeLLM.complete output
    assert any(c[0] == "complete" for c in engine.fake_llm.calls)


def test_switching_embedding_model_reembeds(tmp_path, site):
    _, transport = site
    s = Settings(db_path=str(tmp_path / "e.db"))
    eng = Engine(s, llm=None, embedder=FakeEmbedder(), transport=transport)
    asyncio.run(eng.add(["http://uni.test/"]))
    eng.store.close()

    class Other(FakeEmbedder):
        name = "fake:other"

    eng2 = Engine(s, llm=None, embedder=Other(), transport=transport)
    assert len(eng2.store.chunks_missing_embeddings()) > 0
    eng2.store.close()


# ------------------------------------------------------------------ API
def test_http_api(engine):
    from fastapi.testclient import TestClient

    from webrag.server import create_app

    client = TestClient(create_app(engine))
    assert client.get("/").status_code == 200
    r = client.post("/api/sources", json={"urls": ["http://uni.test/x"]})
    assert r.status_code == 200 and r.json()[0]["status"] == "new"
    assert client.post("/api/sources", json={"urls": ["ftp://nope"]}).status_code == 400

    sources = client.get("/api/sources").json()
    assert len(sources) == 1 and sources[0]["n_chunks"] > 0

    with client.stream("POST", "/api/ask", json={"question": "What is the admission fee?"}) as resp:
        body = "".join(resp.iter_text())
    assert '"type": "sources"' in body and "50,000" in body and '"type": "done"' in body

    sid = sources[0]["id"]
    assert len(client.get(f"/api/sources/{sid}/history").json()["versions"]) == 1
    assert client.delete(f"/api/sources/{sid}").status_code == 200
    assert client.get("/api/sources").json() == []


def test_embedding_outage_degrades_to_keywords(tmp_path, site):
    _, transport = site

    class Broken(FakeEmbedder):
        def embed(self, texts, kind="document"):
            raise ConnectionError("offline")

    eng = Engine(Settings(db_path=str(tmp_path / "b.db")), llm=FakeLLM(), embedder=Broken(), transport=transport)
    [ev] = asyncio.run(eng.add(["http://uni.test/"]))
    assert ev.status == "new"
    answer, _ = eng.answer("What is the admission fee?")
    assert "Rs. 50,000" in answer
    eng.store.close()


def test_fuzzy_class_hints_never_wipe_the_page():
    # Regression: Wikipedia's <html> class contains "main-menu", which used to delete the whole document.
    html = ('<html class="vector-feature-main-menu-disabled vector-toc-available"><body class="skin-nav">'
            '<main><h1>Sam</h1><p>' + "Sam is an entrepreneur and investor. " * 20 + '</p>'
            '<div class="sidebar-menu"><a href="/x">Menu link</a></div></main></body></html>')
    page = extract(html, "http://x/")
    assert "entrepreneur" in page.text
    assert "Menu link" not in page.text


def test_empty_page_is_reported_as_error(tmp_path):
    transport = httpx.MockTransport(lambda r: httpx.Response(200, text="<html><body><div id=app></div></body></html>",
                                                             headers={"content-type": "text/html"}))
    eng = Engine(Settings(db_path=str(tmp_path / "z.db")), llm=None, embedder=None, transport=transport)
    [ev] = asyncio.run(eng.add(["http://spa.test/"]))
    assert ev.status == "error" and "No readable text" in ev.error
    eng.store.close()
