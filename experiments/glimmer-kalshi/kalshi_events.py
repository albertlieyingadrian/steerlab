"""Fetch the settled Kalshi events behind the test set, from Kalshi's public API.

The test rows carry series titles, not tickers. kalshi/series.json (GET /series) maps each title
to its series tickers (one title such as "BTTS" covers 129 leagues); for every matching ticker we
page /events?series_ticker=...&status=settled&min_close_ts=2026-04-01 with nested markets, and save
one compact line per event to kalshi/events.jsonl. No key needed; the public API is read-only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

API = "https://api.elections.kalshi.com/trade-api/v2"
OUT = Path(__file__).resolve().parent / "kalshi"
MIN_CLOSE_TS = 1775001600  # 2026-04-01 00:00 UTC
KEEP = (
    "ticker",
    "yes_sub_title",
    "rules_primary",
    "open_time",
    "close_time",
    "result",
    "title",
)


def fetch_series(ticker: str) -> list[dict]:
    out, cursor = [], ""
    while True:
        params = {
            "series_ticker": ticker,
            "status": "settled",
            "with_nested_markets": "true",
            "limit": 200,
            "min_close_ts": MIN_CLOSE_TS,
        }
        if cursor:
            params["cursor"] = cursor
        for attempt in range(20):
            try:
                r = requests.get(f"{API}/events", params=params, timeout=60)
                if r.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(min(60, 2**attempt))
        r.raise_for_status()
        d = r.json()
        for e in d.get("events", []):
            out.append(
                {
                    "event_ticker": e["event_ticker"],
                    "series_ticker": ticker,
                    "title": e.get("title"),
                    "sub_title": e.get("sub_title"),
                    "category": e.get("category"),
                    "markets": [
                        {k: m.get(k) for k in KEEP} for m in e.get("markets", [])
                    ],
                }
            )
        cursor = d.get("cursor") or ""
        if not cursor or not d.get("events"):
            return out


def main() -> None:
    from concurrent.futures import ThreadPoolExecutor

    series = json.loads((OUT / "series.json").read_text())["series"]
    by_title: dict[str, list[str]] = {}
    for x in series:
        by_title.setdefault(x["title"], []).append(x["ticker"])
    titles = {
        json.loads(line)["state"]["series"]
        for line in (OUT.parent / "data" / "test.jsonl").open()
    }
    tickers = sorted({t for title in titles for t in by_title.get(title, [])})
    print(len(titles), "titles ->", len(tickers), "series tickers", flush=True)
    # Resumable: a ticker's events are written only once all its pages are in, then it is marked done.
    done_file = OUT / "done_series.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    tickers = [t for t in tickers if t not in done]
    print(len(tickers), "left", flush=True)
    n = 0
    with (
        (OUT / "events.jsonl").open("a") as fh,
        done_file.open("a") as dfh,
        ThreadPoolExecutor(4) as pool,
    ):
        for i, (ticker, events) in enumerate(
            zip(tickers, pool.map(fetch_series, tickers))
        ):
            for e in events:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
            fh.flush()
            dfh.write(ticker + "\n")
            dfh.flush()
            n += len(events)
            if i % 200 == 0:
                print(i, n, flush=True)
    print("done", n)


if __name__ == "__main__":
    main()
