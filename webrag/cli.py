"""Command-line interface: `python -m webrag <command>`."""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time

from .engine import NOT_FOUND, Engine

ICONS = {"new": "+ NEW      ", "changed": "~ CHANGED  ", "unchanged": "= UNCHANGED", "error": "! ERROR    "}


def _print_events(events) -> None:
    for e in events:
        extra = " (304 Not Modified)" if e.via_304 else ""
        print(f"{ICONS[e.status]} {e.url}{extra}")
        if e.error:
            print(f"    {e.error}")
        if e.status == "changed":
            print(f"    +{e.added} / -{e.removed} lines, {e.chunks_added} chunks re-indexed, {e.chunks_removed} dropped")
            if e.summary:
                print(f"    Summary: {e.summary}")
            for line in e.diff[:40]:
                print(f"    {line}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="webrag", description="Grounded Q&A over live web pages.")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="index one or more URLs")
    a.add_argument("urls", nargs="+")
    a.add_argument("--depth", type=int, default=0, help="also crawl same-site links N levels deep")

    r = sub.add_parser("refresh", help="re-check all indexed pages and show what changed")
    r.add_argument("--no-summary", action="store_true", help="skip the LLM change summary")

    sub.add_parser("list", help="list indexed sources")

    h = sub.add_parser("history", help="show the version history of a source")
    h.add_argument("source", help="source id or URL")

    rm = sub.add_parser("remove", help="remove a source from the index")
    rm.add_argument("source", help="source id or URL")

    q = sub.add_parser("ask", help="ask a question (omit it for an interactive chat)")
    q.add_argument("question", nargs="*")

    w = sub.add_parser("watch", help="re-check pages periodically")
    w.add_argument("--every", type=float, default=10, help="minutes between checks (default 10)")

    s = sub.add_parser("serve", help="start the web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    args = p.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn

        from .server import create_app

        print(f"WebRAG running at http://{args.host}:{args.port}")
        uvicorn.run(create_app(), host=args.host, port=args.port)
        return 0

    eng = Engine()

    def resolve(ref: str) -> int:
        row = eng.store.source(int(ref)) if ref.isdigit() else eng.store.source_by_url(ref)
        if not row:
            sys.exit(f"No such source: {ref}")
        return row["id"]

    if args.cmd == "add":
        _print_events(asyncio.run(eng.add(args.urls, depth=args.depth)))
    elif args.cmd == "refresh":
        _print_events(asyncio.run(eng.refresh(summarize=not args.no_summary)))
    elif args.cmd == "list":
        rows = eng.store.sources()
        if not rows:
            print("No sources yet. Try: python -m webrag add https://example.com")
        for row in rows:
            checked = time.strftime("%Y-%m-%d %H:%M", time.localtime(row["checked_at"])) if row["checked_at"] else "-"
            print(f"[{row['id']}] {row['title'] or row['url']}\n     {row['url']}\n"
                  f"     {row['n_chunks']} chunks | {row['n_versions']} versions | checked {checked} | {row['status']}")
    elif args.cmd == "history":
        for v in eng.history(resolve(args.source)):
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(v["fetched_at"]))
            print(f"--- version {v['id']} @ {when}" + (" (first seen)" if v["initial"] else f"  +{v['added']}/-{v['removed']}"))
            if v["summary"]:
                print(f"    {v['summary']}")
            for line in v["diff"][:30]:
                print(f"    {line}")
    elif args.cmd == "remove":
        eng.remove(resolve(args.source))
        print("Removed.")
    elif args.cmd == "ask":
        st = eng.status()
        print(f"[{st['llm'] or 'no LLM'} | {st['retrieval']} | {st['sources']} sources]")
        history: list[dict] = []
        question = " ".join(args.question)
        interactive = not question
        while True:
            if interactive:
                try:
                    question = input("\nyou> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not question or question in {"exit", "quit"}:
                    break
            answer, sources = [], []
            print()
            for ev in eng.ask(question, history):
                if ev["type"] == "sources":
                    sources = ev["sources"]
                elif ev["type"] == "token":
                    answer.append(ev["text"])
                    print(ev["text"], end="", flush=True)
                elif ev["type"] == "error":
                    print(f"\n[error] {ev['message']}")
            print()
            text = "".join(answer)
            cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
            if NOT_FOUND in text and not cited:
                sources = []
            elif cited:
                sources = [s_ for s_ in sources if s_["n"] in cited]
            for s_ in sources:
                print(f"  [{s_['n']}] {s_['title']}{' - ' + s_['section'] if s_['section'] else ''}  <{s_['url']}>")
            history += [{"role": "user", "content": question}, {"role": "assistant", "content": "".join(answer)}]
            if not interactive:
                break
    elif args.cmd == "watch":
        print(f"Watching {len(eng.store.sources())} sources every {args.every:g} min. Ctrl+C to stop.")
        try:
            while True:
                events = asyncio.run(eng.refresh())
                stamp = time.strftime("%H:%M:%S")
                changed = [e for e in events if e.status in ("changed", "error")]
                print(f"[{stamp}] checked {len(events)}, {len(changed)} changed/errored")
                _print_events(changed)
                time.sleep(args.every * 60)
        except KeyboardInterrupt:
            pass
    return 0
