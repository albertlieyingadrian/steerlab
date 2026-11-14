"""Fee-aware paper allocation on the test set, following the cookbook README's rule.

Stages (each cached in kalshi/):
  fill    events listed with no nested markets (Kalshi moved them to its historical archive): fetch their markets
          from /historical/markets into kalshi/markets_hist.jsonl
  map     test questions -> Kalshi market tickers (events.jsonl from kalshi_events.py)
            choice: one contract per option, matched on the market's yes_sub_title
            noul:   the market whose rules_primary is the question text after "Will the following resolve Yes? "
  quotes  last completed hourly candle at or before open + 2h, before close (public candlesticks API)
  run     allocate $1,000 per model from results/test_probs.jsonl

Rule: a contract is quote-eligible with a two-sided book and spread <= 10 cents. Bet the side whose probability
exceeds 50% when it beats ask + taker fee by at least 5 cents. Fee: 0.07 * C * P * (1 - P) on a 100-contract
reference order, rounded up to the cent. Stake proportional to (p - 0.5); fractional contracts; static allocation.
90% interval: 5,000 event-bootstrap draws.

    python backtest.py fill | map | quotes | run [--prefix smoke_]
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
K = HERE / "kalshi"
API = "https://api.elections.kalshi.com/trade-api/v2"
PREFIX = "Will the following resolve Yes? "
ENTRY_DELAY_S = 2 * 3600
MAX_SPREAD = 0.10
EDGE = 0.05
BANKROLL = 1000.0


def ts(s: str) -> int:
    return int(datetime.fromisoformat(s).timestamp())


def norm(s: str | None) -> str:
    return " ".join((s or "").split())


_LOCK = threading.Lock()
_NEXT = [0.0]
MIN_INTERVAL_S = 0.2  # Kalshi's public API answers 429 above a few requests per second


def _throttle() -> None:
    with _LOCK:
        now = time.monotonic()
        wait = _NEXT[0] - now
        _NEXT[0] = max(now, _NEXT[0]) + MIN_INTERVAL_S
    if wait > 0:
        time.sleep(wait)


def get(url: str, params: dict) -> dict | None:
    for attempt in range(30):
        _throttle()
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                return None
            if resp.status_code == 429:
                time.sleep(1.0)
                continue
        except requests.RequestException:
            pass
        time.sleep(min(30, 2 ** min(attempt, 5)))
    return None


def hist_markets(event_ticker: str) -> dict:
    out, cursor = [], ""
    while True:
        params = {"event_ticker": event_ticker, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        d = get(f"{API}/historical/markets", params) or {}
        out += [
            {
                k: m.get(k)
                for k in (
                    "ticker",
                    "yes_sub_title",
                    "rules_primary",
                    "open_time",
                    "close_time",
                    "result",
                    "title",
                )
            }
            for m in d.get("markets", [])
        ]
        cursor = d.get("cursor") or ""
        if not cursor or not d.get("markets"):
            return {"event_ticker": event_ticker, "markets": out}


def do_fill() -> None:
    need = (K / "need_markets.txt").read_text().split()
    path = K / "markets_hist.jsonl"
    done = (
        {json.loads(line)["event_ticker"] for line in path.open()}
        if path.exists()
        else set()
    )
    todo = [e for e in need if e not in done]
    print(len(todo), "events to fill")
    with path.open("a") as fh, ThreadPoolExecutor(4) as pool:
        for i, rec in enumerate(pool.map(hist_markets, todo)):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 500 == 0:
                print(i, flush=True)


def do_map() -> None:
    hist = {}
    if (K / "markets_hist.jsonl").exists():
        for line in (K / "markets_hist.jsonl").open():
            rec = json.loads(line)
            hist[rec["event_ticker"]] = [dict(m, hist=True) for m in rec["markets"]]
    events: dict[tuple[str, str], list[dict]] = {}
    for line in (K / "events.jsonl").open():
        e = json.loads(line)
        if not e["markets"]:
            e["markets"] = hist.get(e["event_ticker"], [])
        events.setdefault((norm(e["title"]), norm(e["sub_title"])), []).append(e)
    by_title: dict[str, list[dict]] = {}
    for (t, _), es in events.items():
        by_title.setdefault(t, []).extend(es)
    out, stats = (
        [],
        {
            "rows": 0,
            "rows_matched": 0,
            "contracts": 0,
            "contracts_mapped": 0,
            "ambiguous": 0,
        },
    )
    for row_i, line in enumerate((HERE / "data" / "test.jsonl").open()):
        row = json.loads(line)
        st = row["state"]
        stats["rows"] += 1
        cands = events.get(
            (norm(st["event"]), norm(st.get("event_detail"))), []
        ) or by_title.get(norm(st["event"]), [])
        # Prefer the event whose first market opened on the row's as_of date.
        same_day = [
            e
            for e in cands
            if e["markets"]
            and min(m["open_time"] for m in e["markets"])[:10] == st["as_of"]
        ]
        cands = same_day or cands
        if not cands:
            continue
        stats["rows_matched"] += 1
        markets = [
            dict(m, series_ticker=e["series_ticker"])
            for e in cands
            for m in e["markets"]
        ]
        for qid, q in row["questions"].items():
            if q["type"] == "choice":
                for opt_i, key in enumerate(sorted(q["criteria"])):
                    stats["contracts"] += 1
                    hits = [m for m in markets if norm(m["yes_sub_title"]) == norm(key)]
                    if len(hits) != 1:
                        stats["ambiguous"] += len(hits) > 1
                        continue
                    stats["contracts_mapped"] += 1
                    out.append(
                        {
                            "row": row_i,
                            "qid": qid,
                            "opt": opt_i,
                            "won": str(q["label"]) == key,
                            **pick(hits[0]),
                        }
                    )
            else:
                stats["contracts"] += 1
                text = norm(q["instructions"].removeprefix(PREFIX))
                hits = [m for m in markets if norm(m["rules_primary"]) == text]
                if len(hits) != 1:
                    stats["ambiguous"] += len(hits) > 1
                    continue
                stats["contracts_mapped"] += 1
                out.append(
                    {
                        "row": row_i,
                        "qid": qid,
                        "opt": 0,
                        "won": bool(q["label"]),
                        **pick(hits[0]),
                    }
                )
    with (K / "map.jsonl").open("w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    agree = sum((r["result"] == "yes") == r["won"] for r in out)
    stats["label_agrees_with_kalshi_result"] = f"{agree}/{len(out)}"
    (K / "map_stats.json").write_text(json.dumps(stats, indent=1))
    print(stats)


def pick(m: dict) -> dict:
    return {
        "ticker": m["ticker"],
        "series_ticker": m["series_ticker"],
        "open_time": m["open_time"],
        "close_time": m["close_time"],
        "result": m["result"],
        "hist": bool(m.get("hist")),
    }


def candle(r: dict) -> dict:
    entry = ts(r["open_time"]) + ENTRY_DELAY_S
    end = min(entry, ts(r["close_time"]))
    params = {"start_ts": ts(r["open_time"]), "end_ts": end, "period_interval": 60}
    live = f"{API}/series/{r['series_ticker']}/markets/{r['ticker']}/candlesticks"
    hist = f"{API}/historical/markets/{r['ticker']}/candlesticks"
    # Settled markets move to Kalshi's historical archive; try the likely home first, then the other.
    d = None
    for url in (hist, live) if r.get("hist") else (live, hist):
        d = get(url, params)
        if d is not None:
            break
    if d is None:
        return {"ticker": r["ticker"], "error": "fetch"}
    cs = [c for c in d.get("candlesticks", []) if c["end_period_ts"] <= end]
    if not cs:
        return {"ticker": r["ticker"], "bid": None, "ask": None}
    c = cs[-1]
    # Live markets report close_dollars; the historical archive reports close.
    bid = c.get("yes_bid", {}).get("close_dollars", c.get("yes_bid", {}).get("close"))
    ask = c.get("yes_ask", {}).get("close_dollars", c.get("yes_ask", {}).get("close"))
    return {
        "ticker": r["ticker"],
        "bid": float(bid) if bid else None,
        "ask": float(ask) if ask else None,
        "candle_end": c["end_period_ts"],
    }


def do_quotes() -> None:
    rows = [json.loads(line) for line in (K / "map.jsonl").open()]
    path = K / "quotes.jsonl"
    done = (
        {json.loads(line)["ticker"] for line in path.open()} if path.exists() else set()
    )
    todo = list({r["ticker"]: r for r in rows if r["ticker"] not in done}.values())
    print(len(todo), "markets to fetch")
    with path.open("a", buffering=1) as fh, ThreadPoolExecutor(12) as pool:
        for i, q in enumerate(pool.map(candle, todo)):
            fh.write(json.dumps(q) + "\n")
            if i % 1000 == 0:
                print(i, flush=True)


def fee_per_contract(p: float) -> float:
    return math.ceil(0.07 * 100 * p * (1 - p) * 100 - 1e-9) / 100 / 100


def do_run(prefix: str) -> None:
    quotes = {}
    for line in (K / "quotes.jsonl").open():
        q = json.loads(line)
        quotes[q["ticker"]] = q
    probs = {
        (r["row"], r["qid"]): r
        for r in map(
            json.loads, (HERE / "results" / f"{prefix}test_probs.jsonl").open()
        )
    }
    contracts = [json.loads(line) for line in (K / "map.jsonl").open()]
    eligible = []
    for c in contracts:
        q = quotes.get(c["ticker"], {})
        bid, ask = q.get("bid"), q.get("ask")
        if (
            bid is None
            or ask is None
            or not (0 < bid < ask < 1)
            or ask - bid > MAX_SPREAD
        ):
            continue
        if (c["row"], c["qid"]) not in probs:
            continue
        eligible.append({**c, "bid": bid, "ask": ask})
    models = [
        k for k in next(iter(probs.values())) if k.endswith(("_raw", "_calibrated"))
    ]
    unseen_start = 8765 - 765
    out = {"eligible_contracts": len(eligible), "models": {}}
    for model in models:
        for subset in ("full", "unseen"):
            bets = []
            for c in eligible:
                if subset == "unseen" and c["row"] < unseen_start:
                    continue
                pv = probs[(c["row"], c["qid"])][model]
                p_yes = (
                    pv[c["opt"]]
                    if probs[(c["row"], c["qid"])]["type"] == "choice"
                    else pv[0]
                )
                for side, p, price, wins in (
                    ("yes", p_yes, c["ask"], c["won"]),
                    ("no", 1 - p_yes, 1 - c["bid"], not c["won"]),
                ):
                    cost = price + fee_per_contract(price)
                    if p > 0.5 and p >= cost + EDGE:
                        bets.append(
                            {"row": c["row"], "w": p - 0.5, "cost": cost, "wins": wins}
                        )
            out["models"].setdefault(model, {})[subset] = simulate(bets)
    (HERE / "results" / f"{prefix}backtest.json").write_text(json.dumps(out, indent=1))
    print("eligible contracts", len(eligible))
    for model, d in out["models"].items():
        for subset, s in d.items():
            print(
                f"{model:18s} {subset:6s} events {s['events']:5d} bets {s['bets']:5d} end ${s['end']:.0f} "
                f"90% [{s['ci90'][0]:.0f}, {s['ci90'][1]:.0f}]"
            )


def simulate(bets: list[dict], draws: int = 5000) -> dict:
    if not bets:
        return {"events": 0, "bets": 0, "end": BANKROLL, "ci90": [BANKROLL, BANKROLL]}
    w = np.array([b["w"] for b in bets])
    mult = np.array(
        [(1.0 / b["cost"]) if b["wins"] else 0.0 for b in bets]
    )  # payout per dollar staked
    rows = np.array([b["row"] for b in bets])
    end = float(BANKROLL * (w * mult).sum() / w.sum())
    uniq, inv = np.unique(rows, return_inverse=True)
    ew = np.bincount(inv, weights=w)
    ewm = np.bincount(inv, weights=w * mult)
    rng = np.random.default_rng(0)
    sims = []
    for _ in range(draws):
        s = rng.integers(0, len(uniq), len(uniq))
        sims.append(BANKROLL * ewm[s].sum() / ew[s].sum())
    return {
        "events": len(uniq),
        "bets": len(bets),
        "end": end,
        "ci90": [float(np.percentile(sims, 5)), float(np.percentile(sims, 95))],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fill", "map", "quotes", "run"])
    ap.add_argument("--prefix", default="")
    a = ap.parse_args()
    {
        "fill": do_fill,
        "map": do_map,
        "quotes": do_quotes,
        "run": lambda: do_run(a.prefix),
    }[a.stage]()
