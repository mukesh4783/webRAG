"""Runtime configuration, resolved from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Tiny .env loader: KEY=VALUE lines, no overriding of variables already set."""
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _ollama_url(host: str) -> str:
    """Accept OLLAMA_HOST in Ollama's own formats: '127.0.0.1:11434', '0.0.0.0', 'http://host:port'."""
    if "://" not in host:
        host = "http://" + host
    scheme, _, rest = host.partition("://")
    rest = rest.rstrip("/").replace("0.0.0.0", "127.0.0.1")
    if ":" not in rest:
        rest += ":11434"
    return f"{scheme}://{rest}"


@dataclass
class Settings:
    db_path: str = "webrag.db"

    # Ollama server and models
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "gemma4:31b-cloud"    # answers, follow-up rewriting, change summaries
    embed_model: str = "nomic-embed-text"     # semantic search

    # "ollama" (default) | "none"  — none = retrieval-only / keyword-only modes
    llm: str = "ollama"
    embed: str = "ollama"

    # Retrieval / chunking
    top_k: int = 6
    chunk_chars: int = 1200

    # Fetching
    user_agent: str = "WebRAG/2.0 (+https://github.com/mukesh4783/webRAG)"
    max_pages: int = 25
    fetch_timeout: float = 20.0

    # Uploaded documents (PDF, Word, PowerPoint, text)
    max_upload_mb: float = 50.0

    # Background re-check interval for `serve` (0 = disabled)
    watch_minutes: float = 0.0

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            db_path=_env("WEBRAG_DB", "webrag.db"),
            ollama_url=_ollama_url(_env("OLLAMA_HOST", "127.0.0.1:11434")),
            ollama_model=_env("WEBRAG_MODEL", "gemma4:31b-cloud"),
            embed_model=_env("WEBRAG_EMBED_MODEL", "nomic-embed-text"),
            llm=_env("WEBRAG_LLM", "ollama").lower(),
            embed=_env("WEBRAG_EMBED", "ollama").lower(),
            top_k=int(_env("WEBRAG_TOP_K", "6")),
            chunk_chars=int(_env("WEBRAG_CHUNK_CHARS", "1200")),
            max_pages=int(_env("WEBRAG_MAX_PAGES", "25")),
            watch_minutes=float(_env("WEBRAG_WATCH_MINUTES", "0")),
            max_upload_mb=float(_env("WEBRAG_MAX_UPLOAD_MB", "50")),
        )
