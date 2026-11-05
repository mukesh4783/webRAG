"""A single SQLite file holds everything: sources, their version history, chunks,
a full-text (BM25) index and the embedding vectors. No external vector DB needed."""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS sources (
    id            INTEGER PRIMARY KEY,
    url           TEXT UNIQUE NOT NULL,
    title         TEXT,
    etag          TEXT,
    last_modified TEXT,
    content_hash  TEXT,
    added_at      REAL NOT NULL,
    checked_at    REAL,
    changed_at    REAL,
    status        TEXT DEFAULT 'ok',      -- ok | error
    error         TEXT
);

CREATE TABLE IF NOT EXISTS versions (
    id           INTEGER PRIMARY KEY,
    source_id    INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    content_hash TEXT NOT NULL,
    text         TEXT NOT NULL,
    fetched_at   REAL NOT NULL,
    summary      TEXT                     -- plain-English "what changed", if generated
);

CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY,
    source_id  INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    chunk_hash TEXT NOT NULL,
    section    TEXT,
    text       TEXT NOT NULL,
    embedding  BLOB,
    UNIQUE (source_id, chunk_hash)
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    section, text, content='chunks', content_rowid='id', tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, section, text) VALUES (new.id, new.section, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, section, text) VALUES ('delete', old.id, old.section, old.text);
END;
"""


@dataclass
class Hit:
    chunk_id: int
    source_id: int
    url: str
    title: str
    section: str
    text: str
    checked_at: float | None
    changed_at: float | None
    score: float = 0.0


class Store:
    def __init__(self, path: str = "webrag.db"):
        self.path = path
        self._lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.executescript(SCHEMA)
        self._matrix_cache: tuple[int, np.ndarray, list[int]] | None = None
        self._generation = 0

    # ---------- meta ----------
    def get_meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))

    # ---------- sources ----------
    def source_by_url(self, url: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM sources WHERE url=?", (url,)).fetchone()

    def source(self, source_id: int) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()

    def sources(self) -> list[sqlite3.Row]:
        return self.db.execute(
            """SELECT s.*, (SELECT COUNT(*) FROM chunks c WHERE c.source_id=s.id) AS n_chunks,
                      (SELECT COUNT(*) FROM versions v WHERE v.source_id=s.id) AS n_versions
               FROM sources s ORDER BY s.added_at"""
        ).fetchall()

    def upsert_source(self, url: str) -> int:
        with self._lock, self.db:
            row = self.source_by_url(url)
            if row:
                return row["id"]
            cur = self.db.execute("INSERT INTO sources(url, added_at) VALUES (?, ?)", (url, time.time()))
            return cur.lastrowid

    def update_source(self, source_id: int, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock, self.db:
            self.db.execute(f"UPDATE sources SET {cols} WHERE id=?", (*fields.values(), source_id))

    def delete_source(self, source_id: int) -> None:
        with self._lock, self.db:
            self.db.execute("DELETE FROM chunks WHERE source_id=?", (source_id,))
            self.db.execute("DELETE FROM sources WHERE id=?", (source_id,))
        self._invalidate()

    # ---------- versions ----------
    def add_version(self, source_id: int, content_hash: str, text: str) -> int:
        with self._lock, self.db:
            cur = self.db.execute(
                "INSERT INTO versions(source_id, content_hash, text, fetched_at) VALUES (?, ?, ?, ?)",
                (source_id, content_hash, text, time.time()),
            )
            return cur.lastrowid

    def set_version_summary(self, version_id: int, summary: str) -> None:
        with self._lock, self.db:
            self.db.execute("UPDATE versions SET summary=? WHERE id=?", (summary, version_id))

    def versions(self, source_id: int) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM versions WHERE source_id=? ORDER BY fetched_at DESC, id DESC", (source_id,)
        ).fetchall()

    def latest_version(self, source_id: int) -> sqlite3.Row | None:
        rows = self.versions(source_id)
        return rows[0] if rows else None

    # ---------- chunks ----------
    def chunk_hashes(self, source_id: int) -> dict[str, int]:
        rows = self.db.execute("SELECT id, chunk_hash FROM chunks WHERE source_id=?", (source_id,))
        return {r["chunk_hash"]: r["id"] for r in rows}

    def sync_chunks(self, source_id: int, new: list[tuple[str, str, str]]) -> list[int]:
        """Make the stored chunks match `new` [(hash, section, text)]. Returns ids of inserted chunks
        (only those need embedding — unchanged chunks keep their vectors)."""
        existing = self.chunk_hashes(source_id)
        wanted = {h for h, _, _ in new}
        inserted = []
        with self._lock, self.db:
            stale = [cid for h, cid in existing.items() if h not in wanted]
            self.db.executemany("DELETE FROM chunks WHERE id=?", [(c,) for c in stale])
            for h, section, text in new:
                if h in existing:
                    continue
                cur = self.db.execute(
                    "INSERT INTO chunks(source_id, chunk_hash, section, text) VALUES (?, ?, ?, ?)",
                    (source_id, h, section, text),
                )
                inserted.append(cur.lastrowid)
        if stale or inserted:
            self._invalidate()
        return inserted

    def chunks_missing_embeddings(self) -> list[sqlite3.Row]:
        return self.db.execute(
            """SELECT c.id, c.section, c.text, s.title FROM chunks c JOIN sources s ON s.id=c.source_id
               WHERE c.embedding IS NULL"""
        ).fetchall()

    def set_embeddings(self, items: list[tuple[int, np.ndarray]]) -> None:
        with self._lock, self.db:
            self.db.executemany(
                "UPDATE chunks SET embedding=? WHERE id=?",
                [(v.astype(np.float32).tobytes(), cid) for cid, v in items],
            )
        self._invalidate()

    def clear_embeddings(self) -> None:
        with self._lock, self.db:
            self.db.execute("UPDATE chunks SET embedding=NULL")
        self._invalidate()

    # ---------- search ----------
    def _invalidate(self) -> None:
        self._generation += 1

    def _matrix(self) -> tuple[np.ndarray, list[int]]:
        if self._matrix_cache and self._matrix_cache[0] == self._generation:
            return self._matrix_cache[1], self._matrix_cache[2]
        rows = self.db.execute("SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL").fetchall()
        ids = [r["id"] for r in rows]
        if rows:
            m = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
            m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
        else:
            m = np.zeros((0, 0), dtype=np.float32)
        self._matrix_cache = (self._generation, m, ids)
        return m, ids

    def vector_search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        m, ids = self._matrix()
        if not ids:
            return []
        q = query_vec.astype(np.float32)
        q /= np.linalg.norm(q) + 1e-9
        sims = m @ q
        top = np.argsort(-sims)[:k]
        return [(ids[i], float(sims[i])) for i in top]

    def keyword_search(self, fts_query: str, k: int) -> list[tuple[int, float]]:
        if not fts_query:
            return []
        try:
            rows = self.db.execute(
                "SELECT rowid, bm25(chunks_fts, 0.5, 1.0) AS score FROM chunks_fts "
                "WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
                (fts_query, k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(r["rowid"], -r["score"]) for r in rows]

    def hits(self, chunk_ids: list[int]) -> dict[int, Hit]:
        if not chunk_ids:
            return {}
        marks = ",".join("?" * len(chunk_ids))
        rows = self.db.execute(
            f"""SELECT c.id, c.source_id, c.section, c.text, s.url, s.title, s.checked_at, s.changed_at
                FROM chunks c JOIN sources s ON s.id=c.source_id WHERE c.id IN ({marks})""",
            chunk_ids,
        ).fetchall()
        return {
            r["id"]: Hit(r["id"], r["source_id"], r["url"], r["title"] or r["url"], r["section"] or "",
                         r["text"], r["checked_at"], r["changed_at"])
            for r in rows
        }

    def close(self) -> None:
        self.db.close()
