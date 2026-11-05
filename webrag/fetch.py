"""Polite, concurrent fetching with conditional requests and optional same-site crawling."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx


@dataclass
class FetchResult:
    url: str             # final URL after redirects
    requested: str
    status: int
    html: str = ""
    etag: str | None = None
    last_modified: str | None = None
    not_modified: bool = False
    error: str | None = None


class Fetcher:
    def __init__(self, user_agent: str, timeout: float = 20.0, concurrency: int = 6,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.user_agent = user_agent
        self.client = httpx.AsyncClient(
            headers={"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5"},
            timeout=timeout,
            follow_redirects=True,
            transport=transport,
        )
        self.sem = asyncio.Semaphore(concurrency)
        self._robots: dict[str, RobotFileParser | None] = {}

    async def __aenter__(self) -> "Fetcher":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.client.aclose()

    async def allowed(self, url: str) -> bool:
        p = urlparse(url)
        origin = f"{p.scheme}://{p.netloc}"
        if origin not in self._robots:
            rp: RobotFileParser | None = None
            try:
                r = await self.client.get(origin + "/robots.txt")
                if r.status_code == 200:
                    rp = RobotFileParser()
                    rp.parse(r.text.splitlines())
            except httpx.HTTPError:
                pass
            self._robots[origin] = rp
        rp = self._robots[origin]
        return rp is None or rp.can_fetch(self.user_agent, url)

    async def fetch(self, url: str, etag: str | None = None, last_modified: str | None = None) -> FetchResult:
        headers = {}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified
        async with self.sem:
            try:
                r = await self.client.get(url, headers=headers)
            except httpx.HTTPError as e:
                return FetchResult(url, url, 0, error=f"{type(e).__name__}: {e}")
        final = str(r.url)
        if r.status_code == 304:
            return FetchResult(final, url, 304, etag=etag, last_modified=last_modified, not_modified=True)
        if r.status_code >= 400:
            return FetchResult(final, url, r.status_code, error=f"HTTP {r.status_code}")
        ctype = r.headers.get("content-type", "")
        if ctype and "html" not in ctype and "text/plain" not in ctype:
            return FetchResult(final, url, r.status_code, error=f"Unsupported content type: {ctype}")
        return FetchResult(
            final, url, r.status_code, html=r.text,
            etag=r.headers.get("etag"), last_modified=r.headers.get("last-modified"),
        )


def same_site_links(links: list[str], root: str) -> list[str]:
    """Links on the same host that live under the root URL's path."""
    rp = urlparse(root)
    prefix = rp.path.rsplit("/", 1)[0] + "/"
    out = []
    for link in links:
        p = urlparse(link)
        if p.netloc == rp.netloc and p.path.startswith(prefix) and not p.query:
            if not p.path.lower().endswith((".pdf", ".png", ".jpg", ".jpeg", ".gif", ".zip", ".svg", ".css", ".js")):
                out.append(link)
    return out
