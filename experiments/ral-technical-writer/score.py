"""
Held-out evaluation: pull the volume, build the judge pairs and rating items, then score every source.

    python score.py pull      # samples/, judged/, rated/, trained-judge scores and run logs -> ~/.cache/steerlab/ral-writer/vol
    python score.py pairs     # -> pairs/heldout.jsonl and rate/heldout.jsonl, uploaded to the volume
    python score.py score     # -> results/summary.json (every number in the note) and results/per_title.jsonl

Definitions (plan, "Measures"):
- fooling rate: per (judge, pair), the share of the two orders in which the judge names the REAL abstract as the AI
  one (0.5 = cannot tell); a missing or malformed answer, or an unparsed verdict, counts as not fooled. Per source:
  mean over titles. Ensemble: mean over the judges that pass the known-answer bar (base pairs fooling <= 0.30).
- title match: the abstract's bge-base-en-v1.5 embedding is closest to its own title among the 200 held-out titles.
- loop: the most frequent 4-gram covers more than 10% of the text's 4-grams.
Intervals: 95% bootstrap over titles (2,000 resamples, seed 0); differences paired by title.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

LOCAL = C.CACHE / "vol"
MODAL = str(Path(sys.executable).parent / "modal")
JUDGES = {
    "gptoss": "openai/gpt-oss-120b",
    "gemma": "google/gemma-3-27b-it",
    "phi4": "microsoft/phi-4",
    "glm": "zai-org/GLM-4.5-Air-FP8",
}
RATER = "gemma"
EMBED = "BAAI/bge-base-en-v1.5"
SOURCES = {"base": "base_heldout", "base_fs": "basefs_heldout", "llama_fs": "llamafs_heldout", "sft": "sft_heldout",
           "ral": "ral_heldout", "ral1": "ral1_heldout", "frozen": "frozen_heldout", "humanizer": "humanizer_heldout",
           "stealth": "stealth_heldout"}  # fmt: skip
REWRITERS = ("humanizer", "stealth")
BASE_BAR, MISMATCH_BAR = 0.30, 0.40


def modal(*args: str) -> None:
    subprocess.run([MODAL, *args], check=True, env={**os.environ, "MODAL_PROFILE": "credits30"},
                   stdout=subprocess.DEVNULL)  # fmt: skip


def try_get(remote: str, local: Path) -> None:
    local.mkdir(parents=True, exist_ok=True)
    try:
        modal("volume", "get", "--force", "ral-writer", remote, str(local))
    except subprocess.CalledProcessError:
        print(f"pull: {remote} not on the volume")


def pull() -> None:
    for d in ("samples", "judged", "rated", "jscore"):
        try_get(f"/{d}", LOCAL)
    for run in ("ral_s0", "frozen_s0", "pilot_ral"):
        for f in ("log.jsonl", "config.json", "dev"):
            try_get(f"/runs/{run}/{f}", LOCAL / "runs" / run)
    try_get("/judge0/log.json", LOCAL / "judge0")
    try_get("/sft/adapter/log.json", LOCAL / "sft")
    print(f"pulled into {LOCAL}")


def heldout() -> list[dict]:
    return C.read_jsonl(C.CACHE / "heldout.jsonl")


# Post hoc (added after reading the trained arms' rollouts, not in the plan): both RL arms learned to open with a
# markdown header ("**Title:** ...", "**Abstract**"). The "_clean" sources are the same abstracts with that header and
# all markdown bold removed, so a judge must decide on the prose alone.
CLEANED = ("ral", "ral1", "frozen")
SOURCES |= {f"{s}_clean": SOURCES[s] for s in CLEANED}
HEADER_LINE = re.compile(
    r"^\s*(#+\s*)?(\*\*)?\s*(title|abstract|project summary)\s*:?\s*(\*\*)?\s*:?.*$", re.IGNORECASE
)


def clean(text: str | None) -> str | None:
    """Drop leading title/abstract header lines and markdown bold or heading marks; keep the prose."""
    if not text:
        return text
    lines = text.strip().splitlines()
    while lines and (not lines[0].strip() or (HEADER_LINE.match(lines[0]) and len(lines[0].split()) <= 25)):
        m = re.match(
            r"^\s*(#+\s*)?(\*\*)?\s*(abstract|project summary)\s*:?\s*(\*\*)?\s*:?\s*(.*)$", lines[0], re.IGNORECASE
        )
        if m and len(m.group(5).split()) > 3:  # "Abstract: <prose>" keeps the prose
            lines[0] = m.group(5)
            break
        lines.pop(0)
    t = "\n".join(lines)
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t).replace("**", "")
    t = re.sub(r"^\s*#+\s*", "", t, flags=re.MULTILINE)
    return re.sub(r"[ \t]+\n", "\n", t).strip() or None


def load(source: str) -> dict[int, list]:
    p = LOCAL / "samples" / f"{SOURCES[source]}.jsonl"
    if not p.exists():
        return {}
    out: dict[int, list] = {}
    for r in sorted(C.read_jsonl(p), key=lambda r: (r["appl_id"], r.get("i", 0))):
        a = r.get("answer")
        out.setdefault(r["appl_id"], []).append(clean(a) if source.endswith("_clean") else a)
    return out


def pairs(batch: str = "heldout", only: str = "") -> None:
    """Write pairs/<batch>.jsonl and rate/<batch>.jsonl. Batch "heldout" holds every source present plus the
    mismatch and reference items; a later batch ("late") holds only the named sources (comma-separated), so a source
    that finished after the first judging round is judged with the same prompts without re-judging the rest."""
    H = heldout()
    rows, items = [], []
    want = [s for s in only.split(",") if s] or list(SOURCES)
    for s in want:
        src = load(s)
        if not src:
            print(f"pairs: {s} missing")
            continue
        srcmap = {}
        if s in REWRITERS:
            srcmap = {r["appl_id"]: r.get("source") for r in C.read_jsonl(LOCAL / "samples" / f"{SOURCES[s]}.jsonl")}
        for x in H:
            a = (src.get(x["appl_id"]) or [None])[0]
            if a:
                rows.append({"pid": f"{s}|{x['appl_id']}", "title": x["title"], "real": x["abstract"], "other": a})
                items.append({"iid": f"{s}|{x['appl_id']}", "title": x["title"], "text": a,
                              **({"src": srcmap.get(x["appl_id"])} if s in REWRITERS else {})})  # fmt: skip
    if batch == "heldout":
        for i, x in enumerate(H):
            y = H[(i + 1) % len(H)]
            rows.append({"pid": f"mismatch|{x['appl_id']}", "title": x["title"], "real": x["abstract"],
                         "other": y["abstract"]})  # fmt: skip
            items.append({"iid": f"ref|{x['appl_id']}", "title": x["title"], "text": x["abstract"]})
    C.write_jsonl(LOCAL / "pairs" / f"{batch}.jsonl", rows)
    C.write_jsonl(LOCAL / "rate" / f"{batch}.jsonl", items)
    modal("volume", "put", "-f", "ral-writer", str(LOCAL / "pairs" / f"{batch}.jsonl"), f"/pairs/{batch}.jsonl")
    modal("volume", "put", "-f", "ral-writer", str(LOCAL / "rate" / f"{batch}.jsonl"), f"/rate/{batch}.jsonl")
    print(f"{len(rows)} pairs, {len(items)} rating items uploaded")


# ---------------------------------------------------------------- text measures
def words(t: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", t.lower())


def is_loop(t: str) -> bool:
    w = words(t)
    g = [" ".join(w[i : i + 4]) for i in range(len(w) - 3)]
    return bool(g) and Counter(g).most_common(1)[0][1] / len(g) > 0.10


def distinct_n(texts: list[str], n: int = 2, cut: int = 200) -> float:
    grams = []
    for t in texts:
        w = words(t)[:cut]
        grams += [" ".join(w[i : i + n]) for i in range(len(w) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


def opening_share(texts: list[str], k: int = 3) -> float:
    return Counter(" ".join(words(t)[:k]) for t in texts).most_common(1)[0][1] / len(texts)


def terms(t: str) -> set[str]:
    """The text's numbers and technical tokens: anything with a digit, or a capital after the first letter (acronyms
    and gene or protein names such as NLRP3, VEGF-A, HSPGs). Ordinary capitalised words are left out, since a
    rewrite reorders sentences."""
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", t)
    return {w for w in toks if re.search(r"\d", w) or re.search(r"[A-Z]", w[1:])}


def ngram_hits(t: str, index: set, n: int = 8) -> int:
    w = words(t)
    return sum(" ".join(w[i : i + n]) in index for i in range(len(w) - n + 1))


def boot(vals, n=2000, seed=0):
    a = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    m = a[rng.integers(0, len(a), (n, len(a)))].mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def parse_json_scores(raw: str) -> dict:
    m = re.search(r"\{[^{}]*\}", raw or "")
    if not m:
        return {}
    try:
        return {k: float(v) for k, v in json.loads(m.group(0)).items() if isinstance(v, (int, float))}
    except (ValueError, json.JSONDecodeError):
        return {}


def embed(texts: list[str], query: bool = False) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    if not hasattr(embed, "m"):
        embed.m = SentenceTransformer(EMBED, device="mps")
    pre = "Represent this sentence for searching relevant passages: " if query else ""
    return embed.m.encode([pre + t for t in texts], batch_size=32, normalize_embeddings=True, show_progress_bar=False)


def text_stats(ids, texts_by_id, titles_e, refs_e, train_idx, multi=None) -> dict:
    rec: dict = {}
    texts = [texts_by_id.get(a) for a in ids]
    ok = np.array([bool(t) for t in texts])
    have = [t for t in texts if t]
    rec["valid"] = round(float(ok.mean()), 4)
    if not have:
        return rec
    e = embed([t or "" for t in texts])
    sims = e @ titles_e.T
    rank = np.array([(sims[i] > sims[i, i]).sum() + 1 for i in range(len(ids))])
    match = ((rank == 1) & ok).astype(float)
    rec["title_match"] = round(float(match.mean()), 4)
    rec["title_match_ci"] = boot(match)
    rec["title_mrr"] = round(float(np.where(ok, 1 / rank, 0).mean()), 4)
    rec["sim_to_real"] = round(float(np.mean([e[i] @ refs_e[i] for i in range(len(ids)) if ok[i]])), 4)
    he = e[ok]
    cs = he @ he.T
    rec["cross_title_sim"] = round(float((cs.sum() - np.trace(cs)) / (len(he) * (len(he) - 1))), 4)
    rec["distinct2"] = round(distinct_n(have), 4)
    rec["opening_share"] = round(opening_share(have), 4)
    rec["top_opening"] = Counter(" ".join(words(t)[:3]) for t in have).most_common(1)[0][0]
    rec["loop_share"] = round(float(np.mean([is_loop(t) for t in have])), 4)
    rec["words"] = round(float(np.mean([len(t.split()) for t in have])), 1)
    rec["copy8_per_text"] = round(float(np.mean([ngram_hits(t, train_idx) for t in have])), 2)
    if multi:
        ws = []
        for a in ids:
            v = [t for t in (multi.get(a) or []) if t]
            if len(v) >= 2:
                ev = embed(v)
                m = ev @ ev.T
                ws.append((m.sum() - np.trace(m)) / (len(v) * (len(v) - 1)))
        rec["within_title_sim"] = round(float(np.mean(ws)), 4) if ws else None
    return rec


def pair_fooling(rs: list[dict]) -> float:
    """One pair's fooling score from its two verdicts. Log-prob judges: the order-averaged log-odds of naming the real
    abstract as the AI one, x = ((lpA - lpB | real first) + (lpB - lpA | real second)) / 2, which cancels any constant
    preference for a position; fooled = 1 if x > 0, 0.5 if x = 0, else 0. Letter judges: the share of the two
    verdicts that name the real one (an unparsed verdict counts as not fooled)."""
    if all(r.get("lp_a") is not None for r in rs) and {r["order"] for r in rs} == {"real_first", "real_second"}:
        d = {r["order"]: r["lp_a"] - r["lp_b"] for r in rs}
        x = (d["real_first"] - d["real_second"]) / 2
        return 1.0 if x > 0 else 0.5 if x == 0 else 0.0
    return float(np.mean([bool(r["picked_real"]) for r in rs]))


def score() -> None:
    H = heldout()
    ids = [x["appl_id"] for x in H]
    out: dict = {"n_titles": len(H), "sources": {}, "judges": {}}
    verdict: dict = {}
    for j in JUDGES:
        # {j}.jsonl, then later batches {j}_late*.jsonl (sources that finished after the first judging round)
        for p in [LOCAL / "judged" / f"{j}.jsonl", *sorted((LOCAL / "judged").glob(f"{j}_late*.jsonl"))]:
            if p.exists():
                for r in C.read_jsonl(p):
                    verdict.setdefault(j, {}).setdefault(r["pid"], []).append(r)

    def fool(j, s, aid):
        rs = verdict.get(j, {}).get(f"{s}|{aid}")
        if not rs:
            return 0.0
        return pair_fooling(rs)

    def agree(j, s):
        c = []
        for aid in ids:
            picks = [r["picked_real"] for r in verdict.get(j, {}).get(f"{s}|{aid}") or []]
            if len(picks) == 2 and None not in picks:
                c.append(picks[0] == picks[1])
        return round(float(np.mean(c)), 4) if c else None

    for j, v in verdict.items():
        allr = [r for rs in v.values() for r in rs]
        first = [r["pick"] == "A" for r in allr if r["pick"]]
        kb = float(np.mean([fool(j, "base", a) for a in ids]))
        km = float(np.mean([fool(j, "mismatch", a) for a in ids]))
        out["judges"][j] = {"model": JUDGES[j], "verdicts": len(allr), "unparsed": sum(r["pick"] is None for r in allr),
                            "first_position_rate": round(float(np.mean(first)), 4) if first else None,
                            "known_base_fooling": round(kb, 4), "known_mismatch_fooling": round(km, 4),
                            "passes_base_bar": kb <= BASE_BAR, "passes_mismatch_bar": km <= MISMATCH_BAR,
                            "both_orders_agree_base": agree(j, "base")}  # fmt: skip
    ens = [j for j in verdict if out["judges"][j]["passes_base_bar"]]
    out["ensemble"] = ens
    rated: dict = {}
    for p in [LOCAL / "rated" / f"{RATER}.jsonl", *sorted((LOCAL / "rated").glob(f"{RATER}_late*.jsonl"))]:
        if not p.exists():
            continue
        for r in C.read_jsonl(p):
            rated.setdefault(r["iid"], {}).update(parse_json_scores(r["raw"]))
    titles_e = embed([x["title"] for x in H], query=True)
    refs_e = embed([x["abstract"] for x in H])
    train_idx: set = set()
    for x in C.read_jsonl(C.CACHE / "train.jsonl"):
        w = words(x["abstract"])
        train_idx |= {" ".join(w[i : i + 8]) for i in range(len(w) - 7)}
    ref_rec = text_stats(ids, {x["appl_id"]: x["abstract"] for x in H}, titles_e, refs_e, train_idx)
    for k in ("fit", "fluency"):
        vals = [rated.get(f"ref|{a}", {}).get(k) for a in ids]
        vals = [v for v in vals if v is not None]
        if vals:
            ref_rec[k] = round(float(np.mean(vals)), 3)
    out["sources"]["ref"] = ref_rec
    fool_by: dict = {}
    real_len = {x["appl_id"]: len(x["abstract"].split()) for x in H}
    for s in SOURCES:
        multi = load(s)
        if not multi:
            continue
        s0 = {a: (v or [None])[0] for a, v in multi.items()}
        sub = ids if s not in REWRITERS else [a for a in ids if a in s0]
        rec = text_stats(sub, s0, titles_e[[ids.index(a) for a in sub]], refs_e[[ids.index(a) for a in sub]],
                         train_idx, None if s in REWRITERS else multi)  # fmt: skip
        rec["n_titles"] = len(sub)
        rec["judges"] = {j: {"fooling": round(float(np.mean([fool(j, s, a) for a in sub])), 4),
                             "ci": boot([fool(j, s, a) for a in sub]), "both_orders_agree": agree(j, s)}
                         for j in verdict}  # fmt: skip
        if ens:
            ev = [float(np.mean([fool(j, s, a) for j in ens])) for a in sub]
            fool_by[s] = dict(zip(sub, ev))
            rec["ensemble_fooling"] = round(float(np.mean(ev)), 4)
            rec["ensemble_ci"] = boot(ev)
            lm = [fool_by[s][a] for a in sub if s0.get(a) and abs(len(s0[a].split()) / real_len[a] - 1) <= 0.2]
            rec["ensemble_length_matched"] = {"n": len(lm), "fooling": round(float(np.mean(lm)), 4) if lm else None}
        for k in ("fit", "fluency", "keep"):
            vals = [rated.get(f"{s}|{a}", {}).get(k) for a in sub if s0.get(a)]
            vals = [v for v in vals if v is not None]
            if vals:
                rec[k] = round(float(np.mean(vals)), 3)
                rec[f"{k}_ci"] = boot(vals)
        if s in REWRITERS:
            rows = {r["appl_id"]: r for r in C.read_jsonl(LOCAL / "samples" / f"{SOURCES[s]}.jsonl")}
            kept = []
            for a in sub:
                r = rows[a]
                if r.get("source") and r.get("answer"):
                    t0 = terms(r["source"])
                    if t0:
                        kept.append(len(t0 & terms(r["answer"])) / len(t0))
            rec["terms_kept"] = round(float(np.mean(kept)), 4) if kept else None
            rec["unchanged"] = sum(rows[a].get("answer") == rows[a].get("source") for a in sub)
        out["sources"][s] = rec
    diffs = {}
    for a_, b_ in [("ral", "base"), ("ral1", "base"), ("ral", "frozen"), ("ral1", "frozen"), ("sft", "ral"), ("frozen", "base"), ("sft", "base"),
                   ("llama_fs", "base"), ("base_fs", "base"), ("stealth", "base"), ("humanizer", "base"),
                   ("ral", "llama_fs")]:  # fmt: skip
        if a_ in fool_by and b_ in fool_by:
            com = [a for a in fool_by[a_] if a in fool_by[b_]]
            d = [fool_by[a_][a] - fool_by[b_][a] for a in com]
            diffs[f"{a_}-{b_}"] = {"n": len(com), "diff": round(float(np.mean(d)), 4), "ci": boot(d)}
    out["paired_ensemble_diffs"] = diffs
    js = {}
    for p in sorted((LOCAL / "jscore").glob("*.jsonl")) if (LOCAL / "jscore").exists() else []:
        if "_late" in p.stem:
            continue
        by: dict = {}
        rows = C.read_jsonl(p)
        for late in sorted(p.parent.glob(f"{p.stem}_late*.jsonl")):
            rows += C.read_jsonl(late)
        for r in rows:
            by.setdefault(r["iid"].split("|")[0], []).append(float(1 / (1 + np.exp(-r["margin"]))))
        js[p.stem] = {k: round(float(np.mean(v)), 4) for k, v in by.items()}
    out["trained_judges_p_ref"] = js
    # Pangram, side-measure only (never a reward or a selection rule): label counts and mean fraction_ai per source
    pg: dict = {}
    pp = C.RESULTS / "pangram.jsonl"
    if pp.exists():
        name = {v: k for k, v in SOURCES.items() if not k.endswith("_clean")} | {"ref": "ref"}
        for r in C.read_jsonl(pp):
            if ":keytest" in str(r.get("appl_id")):
                continue
            d = pg.setdefault(
                name.get(r["source"], r["source"]), {"n": 0, "Human": 0, "Mixed": 0, "AI": 0, "fraction_ai": 0.0}
            )
            d["n"] += 1
            d[r.get("prediction_short") or "AI"] = d.get(r.get("prediction_short") or "AI", 0) + 1
            d["fraction_ai"] += float(r.get("fraction_ai") or 0)
        for d in pg.values():
            d["fraction_ai"] = round(d["fraction_ai"] / d["n"], 4)
            d["share_human"] = round(d["Human"] / d["n"], 4)
    out["pangram_side_measure"] = pg
    (C.RESULTS / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    C.write_jsonl(C.RESULTS / "per_title.jsonl",
                  [{"source": s, "appl_id": a, "ensemble_fooling": f} for s, v in fool_by.items() for a, f in v.items()])  # fmt: skip
    keys = ("ensemble_fooling", "title_match", "cross_title_sim", "fit", "fluency", "keep", "words", "loop_share")
    print(json.dumps({s: {k: r.get(k) for k in keys} for s, r in out["sources"].items()}, indent=1))
    print(json.dumps(out["judges"], indent=1))
    print(json.dumps(diffs, indent=1))


if __name__ == "__main__":
    {"pull": pull, "pairs": pairs, "score": score}[sys.argv[1]](*sys.argv[2:])
