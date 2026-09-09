# WebRAG -- Grounded Q&A Bot

> Ask questions against a **live set of web pages** instead of the model's training memory.  
> Built with LangChain, Chroma Vector DB, Ollama, and Streamlit.

---

## What It Does

| Feature | Description |
|---|---|
| **Grounded Q&A** | Scrapes and indexes user-supplied URLs; answers are drawn exclusively from that content |
| **Detects Page Changes** | Re-fetches URLs and compares SHA-256 hashes to detect if a page was updated |
| **Freshness Report** | Shows a unified diff of exactly what changed and re-indexes the updated content |

---

## Tech Stack

```
User supplies URLs
       |
       v
 WebBaseLoader (LangChain)          <- Scrapes web pages via HTTP + BeautifulSoup
       |
       v
 RecursiveCharacterTextSplitter     <- Splits long pages into 1000-char chunks (200 overlap)
       |
       v
 OllamaEmbeddings                   <- nomic-embed-text via Ollama (local)
(nomic-embed-text)
       |
       v
 Chroma Vector DB (local disk)      <- Stores and searches embeddings (./chroma_db)
       |
       v
 ChatOllama (gemma4:31b-cloud)      <- Answers question using ONLY retrieved chunks
       |
       v
 Streamlit UI                       <- Browser interface for indexing, chat, and diff view
```

### Key Libraries
| Library | Role |
|---|---|
| `langchain` | Orchestration: loaders, splitters, retrieval chains |
| `langchain-ollama` | LLM (`gemma4:31b-cloud`) and embeddings (`nomic-embed-text`) via Ollama |
| `langchain-community` | `WebBaseLoader` for scraping web pages |
| `chromadb` | Local persistent vector database |
| `beautifulsoup4` | HTML parsing during web scraping |
| `streamlit` | Web application UI |
| `python-dotenv` | Loads environment variables from `.env` |

---

## Setup & Installation

### 1. Prerequisites

Make sure you have the following installed and running:
- **Python 3.10+**
- **Ollama** (download from https://ollama.com/download)

Pull the required Ollama models:
```bash
ollama pull gemma4:31b-cloud
ollama pull nomic-embed-text
```

### 2. Clone / navigate to the project folder
```
cd web_scrap
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Running the App

### Start the local test server (for Feature 2 & 3 testing)
Open a terminal and run:
```bash
python -m http.server 8000
```
> This serves `test_page.html` at **http://localhost:8000/test_page.html**

### Start the Streamlit app
Open a **second terminal** and run:
```bash
python -m streamlit run app.py
```
> The app will be available at **http://localhost:8501**

---

## Testing All Three Features

### Feature 1 -- Grounded Q&A

**Goal:** Prove the bot answers from the indexed page only, not its own memory.

1. Open **http://localhost:8501**
2. In the **"Source URLs"** sidebar, enter:
   ```
   https://en.wikipedia.org/wiki/Python_(programming_language)
   ```
3. Click **Index / Refresh URLs** and wait for indexing to complete.
4. In the chat box, ask:
   > *"When was Python released according to the text?"*

   **Expected:** The bot answers using the Wikipedia text and shows the source URL underneath.

5. Now ask an off-topic question:
   > *"Who is Hugh Jackman?"*

   **Expected:** The bot replies `"I couldn't find that information in the indexed pages."` -- it is fully grounded and refuses to hallucinate.

---

### Feature 2 -- Answers Update When Pages Change

**Goal:** Show that after re-indexing an updated page, the bot gives a new, correct answer.

1. In the sidebar, replace any existing URL with the local test page:
   ```
   http://localhost:8000/test_page.html
   ```
2. Click **Index / Refresh URLs**. The Freshness Report shows **New**.
3. Ask:
   > *"What is the admission fee?"*

   **Expected:** The bot answers **Rs. 50,000**.

4. Open `test_page.html` and change line 12:
   ```html
   <!-- Before -->
   <p>The current admission fee for the 2026 academic year is Rs. 50,000.</p>

   <!-- After -->
   <p>The current admission fee for the 2026 academic year is Rs. 75,000.</p>
   ```
   Save the file (`Ctrl+S`).

5. Back in the Streamlit app, click **Index / Refresh URLs** again.
6. Ask the same question again:
   > *"What is the admission fee?"*

   **Expected:** The bot now answers **Rs. 75,000** -- confirming the answer updated with the page.

---

### Feature 3 -- Freshness Report with Diff

**Goal:** Show the system detects what exactly changed in a page.

> This is tested as part of the same flow as Feature 2 (step 5 above).

After you save the updated `test_page.html` and click **Index / Refresh URLs**:

- The Freshness Report shows **Changed** next to the URL.
- Click **"View Diff"** to expand the diff view.

**Expected diff output:**
```diff
-    <p>The current admission fee for the 2026 academic year is Rs. 50,000.</p>
+    <p>The current admission fee for the 2026 academic year is Rs. 75,000.</p>
```

---

## Project Structure

```
web_scrap/
|
+-- app.py              # Streamlit frontend -- URL input, chat UI, freshness report
+-- webrag.py           # Core logic -- scraping, hashing, vector DB, RAG chain
+-- test_page.html      # Local mock page to test freshness & change detection
+-- requirements.txt    # Python dependencies
+-- .env                # Environment config (not committed to git)
+-- page_states.json    # Auto-generated: stores URL hashes for change detection
+-- chroma_db/          # Auto-generated: local Chroma vector database
```

---

## Notes

- The `chroma_db/` folder and `page_states.json` are auto-generated and can be safely deleted to reset the index.
- If questions fail after re-indexing, restart Streamlit (`Ctrl+C` then `python -m streamlit run app.py`).
- Make sure Ollama is running before starting the app (`ollama serve` or ensure the Ollama desktop app is open).
