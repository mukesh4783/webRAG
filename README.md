# ◎ WebRAG 2 — grounded, change-aware Q&A over live web pages

Point WebRAG at some web pages and ask questions. Answers come **only** from those pages, every claim is
**cited**, and WebRAG keeps watching the pages: when one changes you get a **word-level diff**, a
**plain-English summary of what changed**, and answers that reflect the new content immediately.

```
you> What is the admission fee?
The admission fee for the 2026 academic year is Rs. 75,000 [1].
  [1] Mock University - Welcome to Mock University > Admission Details  <http://localhost:8765/...>
```

## What's new compared to v1

| | v1 | v2 |
|---|---|---|
| Storage | Chroma + `page_states.json` | **One SQLite file**: sources, full version history, chunks, FTS5 index and vectors |
| Re-indexing | Wiped and re-embedded *everything* on every click | **Incremental**: conditional HTTP (`ETag` / `If-Modified-Since` → 304), content hashing, and only *changed chunks* are re-embedded |
| Extraction | Raw page text (menus, footers and all) | **Readable-content extraction** that keeps headings, lists and tables and drops navigation and boilerplate |
| Chunking | Fixed 1000-char windows | **Section-aware chunks** that never straddle headings; each carries its heading path (`Admissions > Fees`) |
| Retrieval | Vector only | **Hybrid**: BM25 (SQLite FTS5) + embeddings, merged with Reciprocal Rank Fusion. Still works with no embeddings at all |
| Answers | Plain text, sources listed underneath | **Streamed** answers with inline `[n]` citations linked to source cards that show the section and when the page was last checked |
| Off-topic questions | Relied on the prompt alone | Strict citation-only prompt, and when retrieval finds nothing it refuses **without** calling the model |
| Follow-ups | None (stateless) | Conversation-aware: follow-ups like "and the deadline?" are rewritten into standalone search queries |
| Change tracking | Last diff only | **Full version history** per page, with diffs between versions and LLM change summaries |
| Freshness | Manual button | Manual, per-page, `webrag watch`, or automatic background re-checks in the server |
| Coverage | Single URLs | Optional **same-site crawl** (depth 1–2, respects `robots.txt`) |
| Models | Ollama via LangChain | Same Ollama models (`gemma4:31b-cloud`, `nomic-embed-text`) called directly over Ollama's HTTP API, with no LangChain; still works with no model (retrieval-only) |
| UI | Streamlit | FastAPI + a no-build single-page app, plus a full **CLI** |
| Tests | — | Offline test suite (mocked HTTP, fake models) |

## Architecture

```
             ┌──────────── webrag/fetch.py ─────────────┐
 URLs ──────▶│ async httpx · conditional GET · robots.txt │──▶ 304? → "unchanged", done
             │ same-site crawl                            │
             └───────────────────┬────────────────────────┘
                                 ▼
                    webrag/extract.py  readable blocks + heading paths
                                 ▼
             sha256(text) same as last version? → "unchanged"
                                 ▼  (changed or new)
     versions table ◀── store new version, unified diff vs previous, LLM "what changed" summary
                                 ▼
                    webrag/chunk.py   section-aware chunks (hashed)
                                 ▼
     webrag/store.py   sync chunks: delete stale · insert new · embed ONLY new chunks
                       SQLite: sources | versions | chunks(+vectors) | chunks_fts (BM25)
                                 ▼
 question ─▶ (follow-up rewrite) ─▶ BM25 ⊕ cosine ─RRF─▶ top-k excerpts ─▶ LLM (stream) ─▶ cited answer
```

| Module | Role |
|---|---|
| `webrag/engine.py` | Orchestration: add / refresh / history / search / ask |
| `webrag/providers.py` | `OllamaLLM` (streaming chat) and `OllamaEmbedder` |
| `webrag/server.py` | FastAPI JSON API + Server-Sent-Events answer stream |
| `webrag/static/` | Web UI (vanilla HTML/CSS/JS) |
| `webrag/cli.py` | `python -m webrag …` |

## Setup

WebRAG runs on a local [Ollama](https://ollama.com) server. No API keys are needed.

```bash
ollama pull nomic-embed-text         # embeddings (semantic search)
ollama pull gemma4:31b-cloud         # answers (cloud model: run `ollama signin` once)
pip install -e ".[dev]"              # or: pip install -r requirements.txt
```

| Role | Default model | Override |
|---|---|---|
| Answers, follow-up rewriting, change summaries | `gemma4:31b-cloud` | `WEBRAG_MODEL` |
| Embeddings | `nomic-embed-text` (with its `search_query:` / `search_document:` task prefixes) | `WEBRAG_EMBED_MODEL` |
| Ollama address | `127.0.0.1:11434` | `OLLAMA_HOST` |

If Ollama is down, WebRAG degrades instead of crashing: keyword (BM25) search keeps working and missing
embeddings are filled in on the next index or refresh. See `.env.example` for all settings.

## Usage

### Web UI
```bash
python -m webrag serve            # http://127.0.0.1:8000
```
Add URLs on the left, ask questions, and click a `[n]` citation to jump to its source card. **↻ Check for
changes** re-checks every page and opens the freshness report with diffs. The 🕘 button on a source shows its
version timeline. Set `WEBRAG_WATCH_MINUTES=10` to re-check automatically.

### CLI
```bash
python -m webrag add https://en.wikipedia.org/wiki/Python_(programming_language)
python -m webrag add https://docs.example.com/guide/ --depth 1     # also crawl same-site links
python -m webrag ask "When was Python 3.13 released?"
python -m webrag ask                                              # interactive chat with follow-ups
python -m webrag list
python -m webrag refresh                                          # what changed?
python -m webrag history 1                                        # version timeline + diffs
python -m webrag watch --every 15
python -m webrag remove 1
```

## Demo: watch an answer change

```bash
cd examples && python -m http.server 8765        # terminal 1
python -m webrag add http://localhost:8765/mock_university.html
python -m webrag ask "What is the admission fee?"        # → Rs. 50,000 [1]
python -m webrag refresh                                 # → UNCHANGED (304 Not Modified)
# edit examples/mock_university.html: Rs. 50,000 → Rs. 75,000, save
python -m webrag refresh                                 # → CHANGED, diff + summary, 1 chunk re-indexed
python -m webrag ask "What is the admission fee?"        # → Rs. 75,000 [1]
python -m webrag ask "Who is Hugh Jackman?"              # → "I couldn't find that in the indexed pages."
```

## Tests

```bash
pytest
```
The suite runs fully offline. It covers extraction, chunking, the full index → ask → change → re-ask cycle,
304 handling, incremental re-embedding, crawling, follow-up rewriting, embedding outages, and the HTTP API.

## Notes
- Everything lives in `webrag.db`. Delete it to start fresh. Changing the embedding model re-embeds automatically.
- Pages are fetched with a static HTTP client. Pages that only render with JavaScript aren't supported yet.
- `EXPLANATION.md`, `RAG_ppt.pptx` and `ragPhoto/` describe the original v1 (LangChain + Chroma + Streamlit) version.
