"""FastAPI app: JSON API + a dependency-free single-page UI with streamed answers (SSE)."""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .engine import Engine

STATIC = Path(__file__).parent / "static"


class AddRequest(BaseModel):
    urls: list[str]
    depth: int = Field(0, ge=0, le=3)


class RefreshRequest(BaseModel):
    ids: list[int] | None = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    history: list[dict] = []


def _source_dict(r) -> dict:
    return {k: r[k] for k in r.keys()}


def create_app(engine: Engine | None = None) -> FastAPI:
    eng = engine or Engine()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        task = None
        if eng.s.watch_minutes > 0:
            async def watch():
                while True:
                    await asyncio.sleep(eng.s.watch_minutes * 60)
                    try:
                        events = await eng.refresh()
                        changed = [e.url for e in events if e.status == "changed"]
                        if changed:
                            print(f"[webrag] watch: changed -> {changed}")
                    except Exception as e:
                        print(f"[webrag] watch error: {e}")
            task = asyncio.create_task(watch())
        yield
        if task:
            task.cancel()

    app = FastAPI(title="WebRAG", version="2.0", lifespan=lifespan)
    app.state.engine = eng
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    def status():
        return {**eng.status(), "watch_minutes": eng.s.watch_minutes}

    @app.get("/api/sources")
    def sources():
        return [_source_dict(r) for r in eng.store.sources()]

    @app.post("/api/sources")
    async def add(req: AddRequest):
        urls = [u.strip() for u in req.urls if u.strip()]
        if not urls:
            raise HTTPException(400, "Provide at least one URL.")
        bad = [u for u in urls if not u.startswith(("http://", "https://"))]
        if bad:
            raise HTTPException(400, f"URLs must start with http:// or https:// — got {bad[0]!r}")
        events = await eng.add(urls, depth=req.depth)
        return [e.to_dict() for e in events]

    @app.post("/api/refresh")
    async def refresh(req: RefreshRequest):
        events = await eng.refresh(req.ids)
        return [e.to_dict() for e in events]

    @app.delete("/api/sources/{source_id}")
    def delete(source_id: int):
        if not eng.store.source(source_id):
            raise HTTPException(404, "No such source")
        eng.remove(source_id)
        return {"ok": True}

    @app.get("/api/sources/{source_id}/history")
    def history(source_id: int):
        row = eng.store.source(source_id)
        if not row:
            raise HTTPException(404, "No such source")
        return {"source": _source_dict(row), "versions": eng.history(source_id)}

    @app.get("/api/search")
    def search(q: str, k: int = 8):
        return [Engine._hit_dict(i + 1, h) for i, h in enumerate(eng.search(q, k))]

    @app.post("/api/ask")
    def ask(req: AskRequest):
        def events():
            for ev in eng.ask(req.question, req.history):
                yield f"data: {json.dumps(ev)}\n\n"
        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app
