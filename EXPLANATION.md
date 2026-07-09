# 📖 webRAG — Full Project Explanation

> A deep-dive into every component, design decision, and feature of this project.

---

## Table of Contents

1. [What is webRAG?](#1-what-is-webrag)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Tech Stack — Every Library Explained](#3-tech-stack--every-library-explained)
4. [The AI Brain — Model & Embeddings](#4-the-ai-brain--model--embeddings)
5. [The LangChain Pipeline — Step by Step](#5-the-langchain-pipeline--step-by-step)
6. [Feature 1 — Web Indexing with Change Detection](#6-feature-1--web-indexing-with-change-detection)
7. [Feature 2 — Grounded Q&A Chat](#7-feature-2--grounded-qa-chat)
8. [Feature 3 — Freshness Report & Diff Viewer](#8-feature-3--freshness-report--diff-viewer)
9. [File-by-File Breakdown](#9-file-by-file-breakdown)
10. [How Data Flows — End to End](#10-how-data-flows--end-to-end)
11. [Environment & Configuration](#11-environment--configuration)

---

## 1. What is webRAG?

**webRAG** stands for **Web-based Retrieval-Augmented Generation**.

It is an AI-powered Q&A bot that answers your questions using the **live content of real websites** — not its own training data. You give it a list of URLs, it fetches and indexes their content, and then you can chat with it and get answers that are 100% grounded in those specific pages.

The key idea is **RAG (Retrieval-Augmented Generation)**:
- Instead of asking a language model to "remember" facts from training, you first **retrieve** the most relevant chunks of content from your documents.
- You then pass only those chunks to the LLM and ask it to generate an answer.
- This makes the model **accurate, up-to-date, and hallucination-resistant** for your specific domain.

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            USER INTERFACE                                │
│                         Streamlit (app.py)                               │
│          Sidebar: URL input & Index button  │  Main: Chat interface      │
└──────────────────────────┬───────────────────────────┬───────────────────┘
                           │                           │
                    index_urls()                 ask_question()
                           │                           │
┌──────────────────────────▼───────────────────────────▼───────────────────┐
│                           CORE LOGIC (webrag.py)                         │
│                                                                          │
│  WebBaseLoader → Hash Check → Text Splitter → ChromaDB (vector store)   │
│                                                                          │
│  ChromaDB Retriever → Prompt Template → LLM (gpt-4o-mini) → Answer      │
└──────────────────────────────────────────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
  Azure AI Models    ChromaDB on Disk   page_states.json
  (LLM + Embeddings)  (Vector Store)   (Change Tracking)
```

---

## 3. Tech Stack — Every Library Explained

### 🎨 Streamlit
**What it is:** A Python library that turns plain Python scripts into interactive web applications with zero HTML/CSS/JS knowledge required.

**Why we use it:** webRAG needs a UI where users can type URLs, click a button to index them, and then chat. Streamlit gives us all of that — a sidebar, a chat widget, spinners, expandable sections — in just ~80 lines of Python. It handles session state (`st.session_state`) so chat history persists across user interactions within a session.

**Key Streamlit features used:**
- `st.chat_input` / `st.chat_message` — the entire chat UI
- `st.sidebar` — URL input panel
- `st.spinner` — loading indicator during indexing and querying
- `st.expander` — collapsible "Sources" and "View Diff" panels
- `st.session_state` — in-memory chat history storage

---

### 🦜 LangChain (+ langchain-community, langchain-openai, langchain-classic, langchain-core)
**What it is:** An open-source framework for building applications powered by language models. It provides composable building blocks — loaders, splitters, vector stores, chains, prompts — so you don't have to wire every AI component from scratch.

**Why we use it:** Without LangChain, you would need to manually fetch web pages, call the embeddings API, manage vector search, format prompts, call the LLM API, and parse results. LangChain provides all of these as interchangeable modules. The `create_retrieval_chain` function alone replaces ~50 lines of custom code.

**Sub-packages used:**
| Package | Role |
|---|---|
| `langchain-community` | `WebBaseLoader` (web scraping) + `Chroma` vector store wrapper |
| `langchain-openai` | `ChatOpenAI` (LLM calls) + `OpenAIEmbeddings` (embedding calls) |
| `langchain-classic` | `create_retrieval_chain`, `create_stuff_documents_chain` (the RAG pipeline) |
| `langchain-core` | `ChatPromptTemplate` (prompt construction) |
| `langchain` (text splitters) | `RecursiveCharacterTextSplitter` (chunking) |

---

### 🗄️ ChromaDB (`chromadb`)
**What it is:** An open-source, embedded vector database. It stores text alongside its numerical vector representation (embedding) and lets you search for the most semantically similar chunks given a query vector.

**Why we use it:** Traditional databases search by keyword match (`WHERE text LIKE '%keyword%'`). ChromaDB searches by **meaning**. If you ask "Who founded the company?", it will find the chunk that says "John Smith started the firm in 2001" even if your exact words don't appear. It runs entirely on disk locally — no server to set up.

**How it's used in this project:**
- `Chroma.from_documents(...)` — creates a new collection from chunked documents and stores their embeddings
- `Chroma(persist_directory=...)` — re-loads an existing collection from disk
- `vectorstore.as_retriever(search_kwargs={"k": 4})` — returns a retriever that finds the top 4 most relevant chunks for any query
- `old_db.delete_collection()` — wipes the old index before re-indexing to prevent stale duplicates

---

### 🌐 BeautifulSoup4 (`beautifulsoup4`)
**What it is:** A Python library for parsing HTML and XML.

**Why we use it:** `WebBaseLoader` (from LangChain) uses BeautifulSoup under the hood to extract the readable text from web pages. When a web page is fetched, the raw response is HTML filled with tags, scripts, and styles. BeautifulSoup strips all of that and gives us just the human-readable text content.

---

### 🔢 Tiktoken (`tiktoken`)
**What it is:** OpenAI's fast byte-pair encoding tokenizer.

**Why we use it:** LangChain's text splitter uses tiktoken to count tokens accurately when splitting documents. This ensures each chunk stays within the token limits of the model — so a chunk never exceeds what the LLM or embedding model can process in one call.

---

### 🔑 Python-dotenv (`python-dotenv`)
**What it is:** A tiny library that reads a `.env` file and loads its key-value pairs into the process's environment variables.

**Why we use it:** API keys must never be hardcoded in source code. The project stores the `GITHUB_TOKEN` / `OPENAI_API_KEY` in a `.env` file that is excluded from git. `load_dotenv()` is called at the very top of `webrag.py` so that `os.environ.get("OPENAI_API_KEY")` works everywhere without any other setup.

---

### 🐍 Standard Library: `hashlib`, `difflib`, `json`, `shutil`, `os`
| Module | Purpose |
|---|---|
| `hashlib` | SHA-256 hashing of page content to detect changes |
| `difflib` | Unified diff generation to show exactly what changed on a page |
| `json` | Reading/writing `page_states.json` (the change-tracking store) |
| `shutil` | File system utilities (used for directory cleanup) |
| `os` | Environment variable access and file path checks |

---

## 4. The AI Brain — Model & Embeddings

### 🤖 The LLM: `gpt-4o-mini` via Azure AI Inference

```python
def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=get_token(),
        base_url="https://models.inference.ai.azure.com",
    )
```

**What it is:** `gpt-4o-mini` is OpenAI's fast, cost-efficient multimodal model — a smaller, faster variant of GPT-4o. It has strong instruction-following ability and is excellent at reading a block of context and generating a precise, grounded answer.

**Why gpt-4o-mini (not GPT-4o or GPT-3.5)?**
- Cheaper per token than GPT-4o — important when context windows include 4 retrieved chunks (~4000 tokens each call)
- Significantly smarter than GPT-3.5 — produces more coherent, faithful answers
- Fast enough for a real-time chat application

**Why Azure AI Inference endpoint?**
The `base_url` is pointed to `https://models.inference.ai.azure.com` — this is **GitHub Models**, which lets you access OpenAI models using a GitHub Personal Access Token (free tier available). This means you don't need an OpenAI billing account; your `GITHUB_TOKEN` is enough. The `langchain-openai` wrapper is compatible because GitHub Models exposes an OpenAI-compatible REST API.

---

### 🔢 The Embedding Model: `text-embedding-3-small`

```python
def get_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=get_token(),
        base_url="https://models.inference.ai.azure.com",
    )
```

**What is an embedding?** An embedding is a list of floating-point numbers (a vector) that represents the **semantic meaning** of a piece of text. Similar texts have vectors that are mathematically close to each other. For example, the embedding for "What is the capital of France?" will be very close to the embedding for "Paris is the capital city of France."

**What is `text-embedding-3-small`?**
OpenAI's third-generation embedding model. It produces 1536-dimensional vectors. It is significantly more accurate than the older `text-embedding-ada-002` model, at the same price point. The "small" variant is chosen because:
- It's faster than `text-embedding-3-large`
- More than accurate enough for document retrieval tasks
- Lower cost per token

**How it's used:**
1. When indexing — every text chunk gets passed to `text-embedding-3-small` → vector is stored in ChromaDB alongside the original text
2. When querying — the user's question is also converted to a vector → ChromaDB finds the stored vectors with the smallest angular distance (cosine similarity) → those chunks are returned as context

---

## 5. The LangChain Pipeline — Step by Step

The RAG pipeline in `ask_question()` has several clearly defined stages:

### Step 1 — Load the Vector Store
```python
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=get_embeddings())
```
ChromaDB is loaded from disk. It contains all previously indexed and embedded text chunks.

---

### Step 2 — Create a Retriever
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```
The retriever is a component that, given a query string, automatically:
1. Embeds the query using `text-embedding-3-small`
2. Runs a cosine similarity search in ChromaDB
3. Returns the top `k=4` most relevant document chunks

These 4 chunks are your "context" — the raw evidence the LLM will use to answer.

---

### Step 3 — Build the Prompt Template
```python
system_prompt = (
    "Answer the question based ONLY on the following context. "
    "If the answer is not contained in the context, say "
    "'I couldn't find that information in the indexed pages.'\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
```
This is a structured message template. The `{context}` placeholder will be filled with the 4 retrieved chunks. The `{input}` placeholder is the user's question. The system instruction constrains the model to only use the provided context — this is what makes the system **grounded** and prevents hallucination.

---

### Step 4 — Create the Document Chain
```python
question_answer_chain = create_stuff_documents_chain(get_llm(), prompt)
```
`create_stuff_documents_chain` is a LangChain utility that:
1. Takes the list of retrieved documents (chunks)
2. **"Stuffs"** them all together into a single string (concatenated with separators)
3. Fills the `{context}` slot in the prompt template
4. Passes the final prompt to the LLM and gets a response

The word "stuff" is LangChain terminology meaning "put all documents directly into the prompt" — as opposed to more advanced strategies like `map_reduce` or `refine` which are used when there's too much content for a single prompt.

---

### Step 5 — Create the Full Retrieval Chain
```python
chain = create_retrieval_chain(retriever, question_answer_chain)
result = chain.invoke({"input": query})
```
`create_retrieval_chain` wraps the retriever and the document chain into one callable:
1. Takes `{"input": user_query}`
2. Runs the retriever to get relevant chunks (the `context`)
3. Passes `input` + `context` to the document chain
4. Returns `{"answer": "...", "context": [Document, ...]}`

The final result contains both the LLM's answer string and the source `Document` objects (which include the source URL in their metadata).

---

## 6. Feature 1 — Web Indexing with Change Detection

**What it does:** Fetches one or more URLs, converts their content to searchable vector embeddings, stores them in ChromaDB, and tracks whether each page has changed since the last indexing run.

**How it works internally (`index_urls` function):**

```
URL List
    │
    ▼
WebBaseLoader.load()       ← Fetches HTML, strips to plain text via BeautifulSoup
    │
    ▼
For each page:
  SHA-256 hash(text)       ← Fingerprint the content
  Compare to stored hash   ← Is this URL new / changed / unchanged?
  Save new hash + text     ← Update page_states.json
    │
    ▼
RecursiveCharacterTextSplitter
  chunk_size=1000          ← Each chunk is ~1000 characters
  chunk_overlap=200        ← 200-char overlap prevents answers being cut at chunk boundaries
    │
    ▼
OpenAIEmbeddings           ← Convert each chunk to a 1536-dim vector
    │
    ▼
Chroma.from_documents()    ← Store vectors + text + metadata on disk
    │
    ▼
Return change report       ← new / unchanged / changed + diff snippet
```

**Why chunk_overlap=200?**
If a key sentence falls exactly at the boundary between two chunks, without overlap it could be lost from both. The 200-character overlap ensures that boundary content appears in at least one chunk.

**Why delete and recreate ChromaDB on re-index?**
If you just add new documents to the existing collection, you'll accumulate duplicate chunks for unchanged pages. By deleting the collection first (`old_db.delete_collection()`), the index always reflects exactly the current state of the provided URLs — no stale content.

### How to Test Feature 1

1. Start the app: `streamlit run app.py`
2. In the sidebar, paste any URL (e.g., `https://en.wikipedia.org/wiki/Python_(programming_language)`)
3. Click **"Index / Refresh URLs"**
4. You will see a **Freshness Report** appear:
   - 🆕 **New** — first time this URL was indexed
5. Click the button again with the same URL:
   - ✅ **Unchanged** — hash matched, no re-embedding needed
6. Try indexing a URL that updates frequently (e.g., a news site). After some time, re-index and you may see:
   - 🔄 **Changed** — new hash detected, a diff will be shown

---

## 7. Feature 2 — Grounded Q&A Chat

**What it does:** Provides a chat interface where the user asks natural-language questions and receives answers drawn exclusively from the previously indexed web pages. Each answer includes expandable "Sources" showing exactly which chunks of which URLs were used.

**How it works:**

1. The user types a question in `st.chat_input`
2. `webrag.ask_question(query)` is called
3. The LangChain retrieval chain runs (as described in Section 5)
4. The LLM produces an answer grounded only in the retrieved context
5. The answer and source documents are displayed in the chat
6. The conversation is persisted in `st.session_state.messages` so history is maintained for the session

**Why is it grounded / hallucination-resistant?**
The system prompt explicitly says: *"Answer the question based ONLY on the following context. If the answer is not contained in the context, say 'I couldn't find that information in the indexed pages.'"*
The LLM cannot invent facts because it's constrained to only synthesize information from the 4 retrieved chunks provided to it.

**Source attribution:**
Every answer includes an expandable **Sources** section that shows:
- The originating URL (`doc.metadata.get('source')`)
- The first 200 characters of the retrieved chunk

This lets you verify exactly why the LLM said what it said.

### How to Test Feature 2

1. Index one or more URLs (Feature 1 prerequisite)
2. In the main chat area, type a question about the indexed content
   - Example: If you indexed a Wikipedia page on Python, ask: *"Who created the Python programming language?"*
3. The bot will reply with a factual, grounded answer
4. Click **"Sources"** under the answer to see which text chunks were used
5. Try asking something NOT on the indexed pages (e.g., a question about a topic from a different site)
   - The bot should respond: *"I couldn't find that information in the indexed pages."*
6. Ask a follow-up question — the chat history is preserved in the same session

---

## 8. Feature 3 — Freshness Report & Diff Viewer

**What it does:** Every time URLs are re-indexed, the app generates a **Freshness Report** that shows whether each page is new, unchanged, or changed. If changed, it shows a **unified diff** — the exact lines that were added or removed from the page since the last indexing.

**How change detection works:**

```python
def compute_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
```

Each page's full text content is hashed using **SHA-256**. This produces a 64-character hex string that is unique to that exact content. If a single character changes anywhere on the page, the hash changes completely. This is stored in `page_states.json` alongside the full text.

On the next indexing run:
- New hash == old hash → **unchanged** (skip diff)
- New hash != old hash → **changed** → generate unified diff between old text and new text

**The diff format:**
```diff
--- Old
+++ New
@@ -10,5 +10,6 @@
 unchanged line
-removed line
+added line
 another unchanged line
```
Lines starting with `-` were removed, lines with `+` were added. Only the first 30 lines of the diff are shown to keep it readable, with a truncation notice if there's more.

**`page_states.json` structure:**
```json
{
  "https://example.com/page": {
    "hash": "e3b0c44298fc1c149afb...",
    "text": "Full page text content..."
  }
}
```

### How to Test Feature 3

1. Index a URL for the first time → see **🆕 New**
2. Click "Index / Refresh URLs" again immediately → see **✅ Unchanged**
3. To simulate a page change (without waiting for a real site to update):
   - Open `page_states.json` in a text editor
   - Find the entry for your URL
   - Manually change a few characters in the `"text"` field and save
   - Re-index that URL → the hash will no longer match → you'll see **🔄 Changed**
   - Click **"View Diff"** to see the highlighted differences
4. To test with a genuinely changing page:
   - Index a live news homepage (e.g., `https://news.ycombinator.com`)
   - Wait a few hours
   - Re-index → new articles will appear as additions in the diff

---

## 9. File-by-File Breakdown

### `webrag.py` — Core Logic Engine

| Function | What it does |
|---|---|
| `get_token()` | Reads `OPENAI_API_KEY` or `GITHUB_TOKEN` from environment |
| `get_llm()` | Returns a `ChatOpenAI` instance pointing to Azure/GitHub Models |
| `get_embeddings()` | Returns an `OpenAIEmbeddings` instance for `text-embedding-3-small` |
| `load_states()` | Reads `page_states.json` from disk into a Python dict |
| `save_states(states)` | Writes the dict back to `page_states.json` |
| `compute_hash(text)` | Returns SHA-256 hex digest of a string |
| `index_urls(urls)` | Full indexing pipeline: fetch → hash check → split → embed → store |
| `ask_question(query)` | Full RAG pipeline: retrieve → prompt → LLM → return answer + sources |

---

### `app.py` — Streamlit UI Layer

| Section | What it does |
|---|---|
| `st.set_page_config` | Sets browser tab title, favicon, and wide layout |
| Sidebar URL input | `st.text_area` for multi-URL input |
| "Index / Refresh" button | Calls `webrag.index_urls()`, renders the Freshness Report |
| Chat message loop | Renders all past messages from `st.session_state` |
| `st.chat_input` | Captures the user's question and triggers `webrag.ask_question()` |
| Sources expander | Shows which document chunks were retrieved for each answer |

---

### `requirements.txt` — Dependencies

```
streamlit         → Web UI framework
langchain         → Core RAG framework + text splitters
langchain-openai  → OpenAI LLM + embedding wrappers
langchain-community → WebBaseLoader + Chroma integration
chromadb          → Local vector database
beautifulsoup4    → HTML parsing (used internally by WebBaseLoader)
tiktoken          → Token counting for text splitting
python-dotenv     → .env file loading
```

---

### `.env` — Secrets (NOT committed to git)

```
GITHUB_TOKEN=your_github_personal_access_token_here
```
or
```
OPENAI_API_KEY=your_openai_key_here
```

---

### `page_states.json` — Auto-generated, NOT committed

Stores the hash and full text of every indexed URL. Used for change detection across sessions.

---

### `chroma_db/` — Auto-generated, NOT committed

The ChromaDB persistence directory. Contains binary files with the vector embeddings and document metadata. Automatically created on first indexing and rebuilt on each re-index.

---

## 10. How Data Flows — End to End

### Indexing Flow

```
User pastes URLs in sidebar
        │
        ▼
WebBaseLoader fetches each URL
        │
        ▼
BeautifulSoup strips HTML → plain text
        │
        ▼
SHA-256 hash computed per page
        │
        ├─ Hash matches stored hash? → "unchanged"
        ├─ Hash is new? → "new"
        └─ Hash differs? → compute unified diff → "changed"
        │
        ▼
RecursiveCharacterTextSplitter
   splits text into ~1000-char chunks with 200-char overlap
        │
        ▼
text-embedding-3-small API called for each chunk
   → returns 1536-dimensional float vector
        │
        ▼
ChromaDB stores {text, vector, metadata} for each chunk
        │
        ▼
page_states.json updated with new hashes
        │
        ▼
Freshness Report rendered in Streamlit
```

### Query Flow

```
User types question in chat input
        │
        ▼
text-embedding-3-small embeds the question → query vector
        │
        ▼
ChromaDB cosine similarity search
   → returns top 4 most similar chunks
        │
        ▼
ChatPromptTemplate filled:
   system: "Answer ONLY from context..." + [4 chunks]
   human:  [user question]
        │
        ▼
gpt-4o-mini generates answer from context
        │
        ▼
Answer + source documents returned to Streamlit
        │
        ▼
Chat message rendered with expandable Sources panel
```

---

## 11. Environment & Configuration

### Required Setup

1. **Create a `.env` file** in the project root:
   ```
   GITHUB_TOKEN=ghp_your_personal_access_token
   ```
   Get a GitHub PAT from: `https://github.com/settings/tokens`
   Enable access to **GitHub Models** (free tier, no billing needed).

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```
   The app opens at `http://localhost:8501`

### Key Configuration Constants (in `webrag.py`)

| Constant | Value | Purpose |
|---|---|---|
| `STATE_FILE` | `"page_states.json"` | Path to the change-tracking store |
| `CHROMA_DIR` | `"./chroma_db"` | Path to the ChromaDB persistence directory |
| `chunk_size` | `1000` | Max characters per text chunk |
| `chunk_overlap` | `200` | Overlap characters between adjacent chunks |
| `k` (retriever) | `4` | Number of chunks retrieved per query |
| LLM model | `gpt-4o-mini` | Language model for answer generation |
| Embedding model | `text-embedding-3-small` | Model for vectorizing text |
| Inference base URL | `https://models.inference.ai.azure.com` | GitHub Models endpoint |

---

*End of explanation. Every function, every library, and every design decision in this project is documented above.*
