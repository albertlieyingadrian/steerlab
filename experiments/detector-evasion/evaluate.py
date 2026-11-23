"""Held-out evaluation: rewrite the 144 held-out sources with every checkpoint, then score them with every ruler.

    python evaluate.py gen [groups...]      # rewrites -> results/eval/gen/<group>.jsonl
    python evaluate.py score [groups...]    # 9B + content (+ NFKC re-score of odd text) -> results/eval/scores.json
    python evaluate.py transfer             # voi-380M, Desklib, Binoculars, Fast-DetectGPT -> eval/transfer.json
    python evaluate.py encoder              # the detector bake-off's ModernBERT-large (Modal profile modal4)
    python evaluate.py judge [groups...]    # fidelity 1-5 and "which did a person write?" -> results/eval/judge.json

Groups: `prompt-only` (untouched model), `best-of-8` (8 untouched rewrites, the one with the highest `full` reward),
and `<run>@<step>` for steps 50, 150, 200 (primary) and 300 (or the run's last saved step) of every run in
configs/arms.json.
Decoding as Episode 3: temperature 0.7, top-p 0.8, top-k 20, 640 new tokens, one rewrite per source, seed 0.
Held-out sources only (96 Qwen3-8B, 48 Qwen3-4B); nothing here chose a setting.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata

import modal

import budget
from common import (
    APP,
    CLAIM_CHECK,
    FIDELITY,
    HERE,
    HUMAN,
    RESULTS,
    REWRITE_PROMPT,
    ZERO_WIDTH,
    human_passages,
    looped,
    odd_chars,
    read_jsonl,
    sources,
    write_jsonl,
)
from modal_app import reward_of

EVAL = RESULTS / "eval"
GEN = EVAL / "gen"
DATA = HERE / "data"
STEPS = (50, 150, 200, 300)  # 200: P1's pre-registered step (Amendment 5)
FULL = {"content": True, "loop": True}


def heldout() -> list[dict]:
    return sources("heldout") + sources("heldout4b")


def run_groups() -> list[str]:
    from train import configs

    out = []
    for c in configs():
        curve = RESULTS / "runs" / c["name"] / "curve.jsonl"
        saved = [json.loads(x)["step"] for x in curve.read_text().splitlines() if x.strip() and
                 json.loads(x).get("checkpoint")] if curve.exists() else list(STEPS)  # fmt: skip
        last = max(saved)
        out += [f"{c['name']}@{s}" for s in sorted({*[s for s in STEPS if s in saved], last}) if s > 0]
    return out


def all_groups() -> list[str]:
    return ["prompt-only", "best-of-8", *run_groups()]


def gen(groups: list[str]) -> None:
    S = heldout()
    prompts = [REWRITE_PROMPT.format(source=s["text"]) for s in S]
    fn = modal.Function.from_name(APP, "rewrite")
    todo = [g for g in groups if not (GEN / f"{g}.jsonl").exists()]
    args = []
    for g in todo:
        if g == "prompt-only":
            args.append(("", prompts, 1))
        elif g == "best-of-8":
            args.append(("", prompts, 8))
        else:
            run, step = g.split("@")
            args.append((f"{run}/step{step}", prompts, 1))
    if not todo:
        return

    def run():
        return list(fn.starmap(args))

    outs = budget.paid("eval-gen", "H100", 0.4 * len(todo), run, gpus=min(len(todo), 6))
    for g, out in zip(todo, outs, strict=True):
        if g == "best-of-8":
            cands = [{"sid": s["sid"], "k": k, **c} for s, cs in zip(S, out, strict=True) for k, c in enumerate(cs)]
            write_jsonl(DATA / "eval" / "best-of-8-candidates.jsonl", cands)
            R = modal.Cls.from_name(APP, "Reward")()
            by = {s["sid"]: s for s in S}
            sc = budget.paid("eval-bo8-score", "L40S", 0.5, R.score.remote,
                             [{"source": by[c["sid"]]["text"], "rewrite": c["text"], "claims": by[c["sid"]]["claims"]}
                              for c in cands])  # fmt: skip
            rows = []
            for s in S:
                idx = [i for i, c in enumerate(cands) if c["sid"] == s["sid"]]
                best = max(idx, key=lambda i: reward_of(sc[i], FULL, looped(cands[i]["text"]),
                                                        not cands[i]["text"].strip()))  # fmt: skip
                rows.append({"group": g, "sid": s["sid"], "text": cands[best]["text"], "finish": cands[best]["finish"],
                             "tokens": cands[best]["tokens"], "pick": cands[best]["k"]})  # fmt: skip
        else:
            rows = [{"group": g, "sid": s["sid"], **o[0]} for s, o in zip(S, out, strict=True)]
        write_jsonl(GEN / f"{g}.jsonl", rows)
        print(f"{g}: {len(rows)} rewrites")


def normalise(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text) if c not in ZERO_WIDTH)


def score(groups: list[str]) -> None:
    path = EVAL / "scores.json"
    scores = json.loads(path.read_text()) if path.exists() else {}
    by = {s["sid"]: s for s in heldout()}
    items, keys, norm_items, norm_keys = [], [], [], []
    for g in groups:
        for r in read_jsonl(GEN / f"{g}.jsonl"):
            k = f"{g}/{r['sid']}"
            if k in scores:
                continue
            items.append({"source": by[r["sid"]]["text"], "rewrite": r["text"], "claims": by[r["sid"]]["claims"]})
            keys.append(k)
            oc = odd_chars(r["text"], by[r["sid"]]["text"])
            if oc["zero_width"] or oc["other_script"] or oc["non_ascii"]:
                norm_items.append(normalise(r["text"]))
                norm_keys.append(k)
    if not items:
        return
    R = modal.Cls.from_name(APP, "Reward")()
    chunks = [items[i : i + 300] for i in range(0, len(items), 300)]

    def run():
        out = [x for c in R.score.map(chunks) for x in c]
        return out, (R.detect.remote(norm_items) if norm_items else [])

    out, norm = budget.paid("eval-score", "L40S", 0.1 * len(chunks), run, gpus=min(len(chunks), 4))
    for k, s in zip(keys, out, strict=True):
        scores[k] = s
    for k, d in zip(norm_keys, norm, strict=True):
        scores[k]["s9_nfkc"] = d["s9"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scores) + "\n")
    print(f"scored {len(keys)} rewrites ({len(norm_keys)} re-scored after NFKC) -> {path}")


def transfer() -> None:
    sys.path.insert(0, str(HERE.parent / "sounds-like-me"))
    from detectors import voi380

    path = EVAL / "transfer.json"
    T = json.loads(path.read_text()) if path.exists() else {}
    texts = {}
    for g in all_groups():
        if (GEN / f"{g}.jsonl").exists():
            texts |= {f"{g}/{r['sid']}": r["text"] for r in read_jsonl(GEN / f"{g}.jsonl")}
    texts |= {f"source/{s['sid']}": s["text"] for s in heldout()}
    todo = [k for k in texts if "voi380" not in T.get(k, {})]
    if todo:
        for k, v in zip(todo, voi380([texts[k] or "." for k in todo]), strict=True):
            T.setdefault(k, {})["voi380"] = round(float(v), 6)
        print(f"voi380: {len(todo)}")
    todo = [k for k in texts if "desklib" not in T.get(k, {})]
    if todo:
        det = modal.Cls.from_name("deslop", "Detectors")()
        chunks = [todo[i : i + 150] for i in range(0, len(todo), 150)]

        def run():
            return list(det.score.map([[texts[k] or "." for k in c] for c in chunks]))

        outs = budget.paid("eval-desklib", "L40S", 0.15 * len(chunks), run, gpus=min(len(chunks), 4))
        for c, vals in zip(chunks, outs, strict=True):
            for k, v in zip(c, vals, strict=True):
                T.setdefault(k, {}).update({f: v[f] for f in ("desklib", "binoculars", "fastdetect")})
        print(f"desklib/binoculars/fastdetect: {len(todo)}")
    path.write_text(json.dumps(T) + "\n")


def encoder() -> None:
    """ModernBERT-large from our detector bake-off (run bb-modernbert-large: full fine-tune on slop-detector data v2,
    4-bucket head; branch feat/slop-detector-backbones, not yet on main), scored by that study's deployed function in
    the modal4 workspace on every held-out rewrite, every source and the 318 human passages (for its 1% threshold).
    Output: expected bucket / 3, higher = more AI. Stored as `modernbert` in results/eval/transfer.json."""
    import os
    import subprocess

    path = EVAL / "transfer.json"
    T = json.loads(path.read_text()) if path.exists() else {}
    texts = {}
    for g in all_groups():
        if (GEN / f"{g}.jsonl").exists():
            texts |= {f"{g}/{r['sid']}": r["text"] for r in read_jsonl(GEN / f"{g}.jsonl")}
    texts |= {f"source/{s['sid']}": s["text"] for s in heldout()}
    texts |= {f"human/{h['source']}/{h['id']}": h["text"] for h in human_passages()}
    todo = [k for k in texts if "modernbert" not in T.get(k, {})]
    if not todo:
        return
    local = DATA / "encoder_texts.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps([texts[k] or "." for k in todo]))
    if os.environ.get("MODAL_PROFILE") != "modal4":
        raise SystemExit("run this step as: MODAL_PROFILE=modal4 python evaluate.py encoder (the model lives there)")
    split = "detector-evasion"
    put = [budget.MODAL_BIN, "volume", "put", "--force", "slop-detector", str(local), f"/bb/evals/{split}.json"]
    subprocess.run(put, check=True, capture_output=True)
    fn = modal.Function.from_name("slop-backbones", "score")
    t0 = __import__("time").time()
    out = fn.remote("bb-modernbert-large", [split])
    sec = __import__("time").time() - t0
    with budget.SPEND.open("a") as fh:
        stamp = __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime())
        row = {
            "time": stamp,
            "job": "eval-modernbert",
            "gpu": "H100",
            "gpus": 1,
            "seconds": round(sec, 1),
            "projected": 0.1,
            "usd": round(sec / 3600 * 4.6, 4),
            "profile": "modal4",
        }
        fh.write(json.dumps(row) + "\n")
    for k, v in zip(todo, out["scores"][split], strict=True):
        T.setdefault(k, {})["modernbert"] = round(float(v), 6)
    path.write_text(json.dumps(T) + "\n")
    print(f"modernbert: {len(todo)} texts in {sec:.0f} s")


def judge(groups: list[str]) -> None:
    """Per rewrite: the held-out content ruler (gemma checks each of the source's six key claims against the rewrite,
    yes/no), the 1-5 fidelity rating, and "which did a person write?" rewrite vs source in both orders."""
    path = EVAL / "judge.json"
    J0 = json.loads(path.read_text()) if path.exists() else {}
    by = {s["sid"]: s for s in heldout()}
    keys, fid, hum, cq = [], [], [], []
    for g in groups:
        for r in read_jsonl(GEN / f"{g}.jsonl"):
            k = f"{g}/{r['sid']}"
            if k in J0:
                continue
            src, rw = by[r["sid"]]["text"], r["text"] or "."
            keys.append(k)
            fid.append(FIDELITY.format(a=src, b=rw))
            hum += [HUMAN.format(a=rw, b=src), HUMAN.format(a=src, b=rw)]
            cq += [CLAIM_CHECK.format(text=rw, claim=c) for c in by[r["sid"]]["claims"]]
    if not keys:
        return
    J = modal.Cls.from_name(APP, "Judge")()

    def run():
        return J.rate.remote(fid), J.ask.remote(hum), J.ask.remote(cq)

    rates, picks, claims = budget.paid("eval-judge", "H100", 0.5 + 0.001 * len(keys), run)
    for i, k in enumerate(keys):
        first, second = picks[2 * i], picks[2 * i + 1]
        a_first, b_second = first["pA"] >= first["pB"], second["pB"] > second["pA"]
        J0[k] = {"fidelity": rates[i], "claims_judge": sum(x["pA"] > x["pB"] for x in claims[6 * i : 6 * i + 6]) / 6,
                 "human_pick": "rewrite" if a_first and b_second else
                 "source" if not a_first and not b_second else "split"}  # fmt: skip
    path.write_text(json.dumps(J0) + "\n")
    print(f"judged {len(keys)} rewrites -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["gen", "score", "transfer", "encoder", "judge"])
    ap.add_argument("groups", nargs="*")
    a = ap.parse_args()
    groups = a.groups or all_groups()
    if a.step == "encoder":
        encoder()
        return
    {"gen": gen, "score": score, "judge": judge}.get(a.step, lambda _g: transfer())(groups)


if __name__ == "__main__":
    main()
