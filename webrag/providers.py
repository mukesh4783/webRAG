"""Model backends, served by a local Ollama instance (https://ollama.com).

* OllamaLLM      — chat model for answers, query rewriting and change summaries (default: gemma4:31b-cloud)
* OllamaEmbedder — embedding model for semantic search (default: nomic-embed-text)

Both talk to Ollama's HTTP API directly with httpx, so no extra SDK is required.
If Ollama is not reachable, WebRAG still works: keyword (BM25) search keeps running and
answers fall back to showing the most relevant passages.
"""

from __future__ import annotations

import json
from typing import Iterator, Protocol

import httpx
import numpy as np

from .config import Settings


class LLM(Protocol):
    name: str

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]: ...

    def complete(self, system: str, prompt: str) -> str: ...


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str], kind: str = "document") -> list[np.ndarray]: ...


class OllamaLLM:
    def __init__(self, model: str, base_url: str, timeout: float = 300.0):
        self.model = model
        self.name = f"ollama:{model}"
        self.client = httpx.Client(base_url=base_url, timeout=httpx.Timeout(timeout, connect=10.0))

    def _payload(self, system: str, messages: list[dict], stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": stream,
        }

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        with self.client.stream("POST", "/api/chat", json=self._payload(system, messages, True)) as r:
            if r.status_code >= 400:
                r.read()
                raise RuntimeError(f"Ollama /api/chat failed ({r.status_code}): {r.text[:300]}")
            for line in r.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if event.get("error"):
                    raise RuntimeError(f"Ollama: {event['error']}")
                piece = (event.get("message") or {}).get("content", "")
                if piece:
                    yield piece
                if event.get("done"):
                    break

    def complete(self, system: str, prompt: str) -> str:
        r = self.client.post("/api/chat", json=self._payload(system, [{"role": "user", "content": prompt}], False))
        r.raise_for_status()
        return ((r.json().get("message") or {}).get("content") or "").strip()


class OllamaEmbedder:
    batch = 32

    def __init__(self, model: str, base_url: str, timeout: float = 120.0):
        self.model = model
        self.name = f"ollama:{model}"
        self.client = httpx.Client(base_url=base_url, timeout=httpx.Timeout(timeout, connect=10.0))
        # nomic-embed-text is trained with task prefixes; using them noticeably improves retrieval.
        self._prefixed = "nomic-embed" in model

    def embed(self, texts: list[str], kind: str = "document") -> list[np.ndarray]:
        if self._prefixed:
            prefix = "search_query: " if kind == "query" else "search_document: "
            texts = [prefix + t for t in texts]
        out: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch):
            r = self.client.post("/api/embed", json={"model": self.model, "input": texts[i : i + self.batch]})
            r.raise_for_status()
            out.extend(np.asarray(v, dtype=np.float32) for v in r.json()["embeddings"])
        return out


def make_llm(s: Settings) -> LLM | None:
    if s.llm == "none":
        return None
    return OllamaLLM(s.ollama_model, s.ollama_url)


def make_embedder(s: Settings) -> Embedder | None:
    if s.embed == "none":
        return None
    return OllamaEmbedder(s.embed_model, s.ollama_url)
