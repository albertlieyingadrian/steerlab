"""Step 3: the hard-negative rounds. Every GPU call is spawned on the deployed app and its call id saved first, so a
dropped laptop loses nothing; rerunning a command resumes from what is on disk.

    python loop.py prep      # freeze the evaluation and interp sets, upload them (hn/evals.jsonl, hn/interp.jsonl)
    python loop.py pilot     # plumbing check on the dev pool only (a 0.6B, 4,000 rows, 1 epoch): not a result
    python loop.py round0    # score the held-out sets and the mining pool with round 0 (b1b-1.7bB)
    python loop.py arm a     # arm a: rounds 1-4 (b, c: 1-3); each round mines with the previous round's detector
    python loop.py seeds     # seed-1 retrains of round 0 and arm a round 1

Arms: a = the K hardest pool pages added as human; b = the same plus an AI mirror of each page, added as generated;
c = K random pool pages added as human. Additions accumulate. Only the mining pool is ever mined.
"""

from __future__ import annotations

import json
import random
import sys
import time

import numpy as np

from hn import DATA, FLAG_CUT, GEN_W, HUMAN_W, RESULTS, ROUND0, ROUNDS, K, read_jsonl, reserve, settle, write_jsonl
from pools import vget, vput

APP = "slop-hard-negatives"
SHARDS = 4


def fn(name: str):
    import modal

    return modal.Function.from_name(APP, name)


def run_path(name: str) -> str:
    return ROUND0 if name == "r0" else f"hn/runs/{name}"


def out_path(name: str) -> str:
    return f"hn/runs/{name}"


# ------------------------------------------------------------------ frozen sets
def prep() -> None:
    """The evaluation file (every held-out set, labels kept locally) and the interp file (dev concept sets)."""
    from common import editlens_split, epoch_split  # slop-detector/common.py

    rows, man = [], {}

    def add(name: str, texts, ys, kinds, ids) -> None:
        man[name] = {"y": [int(y) for y in ys], "kind": [str(k) for k in kinds], "id": [str(i) for i in ids]}
        rows.extend({"set": name, "i": i, "text": str(t)} for i, t in enumerate(texts))

    held = read_jsonl(DATA / "pools" / "heldout.jsonl")[:20000]
    add("web", [r["text"] for r in held], [0] * len(held), ["web"] * len(held), [r["id"] for r in held])
    for split in ("test", "test_enron"):
        df = editlens_split(split)
        ids = df.get("text_id", df.index)
        add(split, df["text"], (df["y"] > 0).astype(int), df["text_type"], ids)
    hd = editlens_split("human_detectors")
    add("hd", hd["text"], hd["label"], np.where(hd["label"] == 0, "human", hd["model"]), hd.index)
    ep = epoch_split()
    add("epoch", ep["text"], ep["label"], ep["condition"], ep["text_id"])
    fr = read_jsonl(DATA / "fresh.jsonl")
    for topic in ("old", "new"):
        rs = [r for r in fr if r["topic"] == topic]
        add(f"fresh_{topic}", [r["text"] for r in rs], [1] * len(rs), [r["model"] for r in rs], [r["key"] for r in rs])
    write_jsonl(DATA / "evals.jsonl", rows)
    (RESULTS / "evalset.json").write_text(json.dumps(man) + "\n")
    print({k: len(v["y"]) for k, v in man.items()}, len(rows))

    # interp: corporate register (judge YES vs NO among 3,000 dev pages) and slop (400 dev pages vs their mirrors)
    from gen import judge

    src = read_jsonl(DATA / "interp_corp_src.jsonl")
    ans = judge([r["text"] for r in src], "corporate", "judge corporate (cached)")
    yes = [r for r in src if ans[r["text"]] == "YES"][:400]
    no = [r for r in src if ans[r["text"]] == "NO"]
    no = random.Random(0).sample(no, min(400, len(no)))
    slop_h = {r["id"]: r for r in read_jsonl(DATA / "interp_slop_src.jsonl")}
    mir = [m for m in read_jsonl(DATA / "interp_slop_mirrors.jsonl") if m["doc_id"] in slop_h]
    irows, iman = [], {"set": [], "y": [], "id": [], "words": []}

    def iadd(s: str, text: str, y: int, i: str) -> None:
        irows.append({"set": s, "text": text})
        iman["set"].append(s)
        iman["y"].append(y)
        iman["id"].append(i)
        iman["words"].append(len(text.split()))

    for r in yes:
        iadd("corp", r["text"], 1, r["id"])
    for r in no:
        iadd("corp", r["text"], 0, r["id"])
    for m in mir:
        iadd("slop", slop_h[m["doc_id"]]["text"], 0, m["doc_id"])
        iadd("slop", m["text"], 1, m["key"])
    write_jsonl(DATA / "interp.jsonl", irows)
    (RESULTS / "interpset.json").write_text(json.dumps(iman) + "\n")
    print(f"interp: corp {len(yes)} yes / {len(no)} no of {len(src)}; slop {len(mir)} pairs")
    vput(DATA / "evals.jsonl", "hn/evals.jsonl")
    vput(DATA / "interp.jsonl", "hn/interp.jsonl")


# ------------------------------------------------------------------ GPU calls (spawned, ids saved, resumable)
def _call(tag: str, spawn, gpu: str, hours: float, n: int = 1):
    """Spawn once (call id saved to results/calls/<tag>.json before waiting), then wait; resume a saved call."""
    import modal

    p = RESULTS / "calls" / f"{tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        cid = json.loads(p.read_text())["call_id"]
        print(f"{tag}: resuming call {cid}", flush=True)
        return modal.FunctionCall.from_id(cid)
    reserve(tag, gpu, hours, n)
    call = spawn()
    p.write_text(json.dumps({"call_id": call.object_id, "t": time.strftime("%Y-%m-%d %H:%M")}) + "\n")
    print(f"{tag}: spawned {call.object_id}", flush=True)
    return call


def score_pool(name: str, src: str = "hn/pools/pool.jsonl", tag: str | None = None) -> np.ndarray:
    """The pool's scores under detector `name`, in pool order (cached in data/pool_scores/)."""
    tag = tag or f"pool_{name}"
    local = DATA / "pool_scores" / f"{tag}.npy"
    if local.exists():
        return np.load(local)
    f = fn("score_pool")
    calls = [
        _call(f"{tag}_{k}", lambda k=k: f.spawn(run_path(name), f"{out_path(name)}/{tag}", k, SHARDS, src), "H100", 1)
        for k in range(SHARDS)
    ]
    secs = [c.get(timeout=None)["seconds"] for c in calls]
    for k, s in enumerate(secs):
        settle(f"{tag}_{k}", "H100", s)
    parts = []
    for k in range(SHARDS):
        dst = DATA / "pool_scores" / f"{tag}_{k}.npy"
        dst.parent.mkdir(parents=True, exist_ok=True)
        vget(f"{out_path(name)}/{tag}/pool_scores_{k}.npy", dst)
        parts.append(np.load(dst))
    out = np.zeros(sum(len(p) for p in parts), dtype=np.float32)
    for k, p in enumerate(parts):
        out[k::SHARDS] = p
    np.save(local, out)
    print(
        f"{tag}: {len(out)} scored, slowest shard {max(secs)} s, {int((out >= FLAG_CUT).sum())} at or above the cut",
        flush=True,
    )
    return out


def fetch_outputs(name: str) -> None:
    (RESULTS / "scores").mkdir(parents=True, exist_ok=True)
    (DATA / "feats").mkdir(parents=True, exist_ok=True)
    vget(f"{out_path(name)}/eval_scores.json", RESULTS / "scores" / f"{name}.json")
    vget(f"{out_path(name)}/feats.npz", DATA / "feats" / f"{name}.npz")
    if name != "r0":  # round 0's validation scores are in the slop-detector note (runs/b1b-1.7bB)
        (RESULTS / "val").mkdir(parents=True, exist_ok=True)
        vget(f"{out_path(name)}/val_scores.json", RESULTS / "val" / f"{name}.json")


def evaluate_round0() -> None:
    if (RESULTS / "scores" / "r0.json").exists():
        return
    c = _call("eval_r0", lambda: fn("evaluate").spawn(ROUND0, out_path("r0")), "H100", 1)
    r = c.get(timeout=None)
    settle("eval_r0", "H100", r["seconds"])
    fetch_outputs("r0")
    (RESULTS / "train").mkdir(parents=True, exist_ok=True)
    (RESULTS / "train" / "r0.json").write_text(json.dumps({"name": "r0", "run": ROUND0, **r}, indent=1) + "\n")


def train(name: str, added: list[str], seed: int = 0, extra: list[str] | None = None, **kw) -> dict:
    res_p = RESULTS / "train" / f"{name}.json"
    if res_p.exists():
        return json.loads(res_p.read_text())
    # worst case: 150-minute timeout x 2 attempts
    tag = f"train_{name}"
    c = _call(tag, lambda: fn("train_round").spawn(name, added, seed, extra or [], **kw), "H100", 5)
    r = c.get(timeout=None)
    # A Modal retry reports only its own attempt. When the call took more wall time than that attempt plus a generous
    # start-up allowance (20 minutes), bill the wall time since spawn instead, so a failed first attempt is counted.
    spawned = time.mktime(
        time.strptime(json.loads((RESULTS / "calls" / f"{tag}.json").read_text())["t"], "%Y-%m-%d %H:%M")
    )
    wall = time.time() - spawned
    settle(tag, "H100", wall if wall > r["total_seconds"] + 1200 + 9000 else r["total_seconds"])
    if "evals" not in kw:
        fetch_outputs(name)
    res_p.parent.mkdir(parents=True, exist_ok=True)
    res_p.write_text(json.dumps(r, indent=1) + "\n")
    print(f"{name}: trained in {r['train_seconds']} s, {r['added_rows']} added rows", flush=True)
    return r


# ------------------------------------------------------------------ the rounds
def human_row(p: dict, src: str) -> dict:
    return {
        "id": p["id"],
        "text": p["text"],
        "target": 0.0,
        "kind": "human",
        "source": src,
        "band": 0,
        "bucket": 0,
        "weight": HUMAN_W,
        "key": f"{p['id']}|{src}",
    }


def mirror_row(m: dict) -> dict:
    return {
        "id": m["doc_id"],
        "text": m["text"],
        "target": 1.0,
        "kind": "mirror",
        "source": "fineweb-mirror",
        "model": m["model"],
        "band": 2,
        "bucket": 3,
        "weight": GEN_W,
        "key": m["key"],
    }


def choose(a: str, r: int, pool: list[dict], added_ids: set[str]) -> dict:
    """Round r's pages for arm a: the K highest-scored pages not yet added (a, b), or K random ones (c)."""
    free = np.array([p["id"] not in added_ids for p in pool])
    prev = "r0" if r == 1 else f"{a}{r - 1}"
    meta: dict = {"arm": a, "round": r, "detector": prev if a != "c" else None}
    if a == "c":
        pick = np.array(sorted(random.Random(f"c{r}").sample(list(np.where(free)[0]), K)))
    else:
        s = score_pool(prev)
        order = np.lexsort((np.arange(len(s)), -s))  # highest score first, ties by pool order
        pick = np.array([i for i in order if free[i]][:K])
        sc = s[pick]
        meta |= {
            "scores": [round(float(x), 5) for x in sc],
            "above_cut": int((sc >= FLAG_CUT).sum()),
            "min_score": float(sc.min()),
            "max_score": float(sc.max()),
            "pool_above_cut": int((s >= FLAG_CUT).sum()),
        }
    meta |= {"ids": [pool[i]["id"] for i in pick], "pool_index": [int(i) for i in pick]}
    return meta


def arm(a: str) -> None:
    pool = read_jsonl(DATA / "pools" / "pool.jsonl")
    added_files: list[str] = []
    added_ids: set[str] = set()
    for r in range(1, ROUNDS[a] + 1):
        name = f"{a}{r}"
        add_p = DATA / "added" / f"{a}_r{r}.jsonl"
        mined_p = RESULTS / "mined" / f"{a}_r{r}.json"
        if not add_p.exists():
            meta = json.loads(mined_p.read_text()) if mined_p.exists() else choose(a, r, pool, added_ids)
            mined_p.parent.mkdir(parents=True, exist_ok=True)
            mined_p.write_text(json.dumps(meta) + "\n")
            pages = [pool[i] for i in meta["pool_index"]]
            rows = [human_row(p, "fineweb-random" if a == "c" else "fineweb-hard") for p in pages]
            if a == "b":
                from gen import mirrors

                mout = DATA / "mirrors" / f"b_r{r}.jsonl"
                want = set(meta["ids"])
                for _ in range(2):  # a rerun skips the mirrors already made, so a second pass fills provider gaps
                    mirrors(pages, mout, f"mirrors b r{r}")
                    got = [m for m in read_jsonl(mout) if m["doc_id"] in want]
                    if len(got) == len(pages):
                        break
                if (
                    len(got) < len(pages) * 0.995
                ):  # never train arm b on a short pair set (b_r1 ran with 1,999 of 2,000)
                    raise SystemExit(f"{name}: only {len(got)} of {len(pages)} mirrors; rerun before training")
                rows += [mirror_row(m) for m in got]
                meta["mirrors"] = len(got)
                mined_p.write_text(json.dumps(meta) + "\n")
            write_jsonl(add_p, rows)
            print(
                f"{name}: {len(pages)} pages"
                + (f", {meta['above_cut']} above the cut" if "above_cut" in meta else "")
                + (f", {meta.get('mirrors')} mirrors" if a == "b" else ""),
                flush=True,
            )
        vput(add_p, f"hn/added/{a}_r{r}.jsonl")
        added_files.append(f"hn/added/{a}_r{r}.jsonl")
        added_ids |= set(json.loads(mined_p.read_text())["ids"])
        train(name, list(added_files))
    print(f"arm {a} done", flush=True)


def seeds() -> None:
    train("r0s1", [], seed=1)
    train("a1s1", ["hn/added/a_r1.jsonl"], seed=1)


def pilot() -> None:
    """Dev pool only. Timing of pool scoring, flagged share on dev, and one tiny training through the launcher."""
    s = score_pool("r0", src="hn/pools/dev.jsonl", tag="pilot_dev_r0")
    dev = read_jsonl(DATA / "pools" / "dev.jsonl")
    print(f"dev flagged at the cut: {int((s >= FLAG_CUT).sum())} of {len(s)}")
    top = np.argsort(-s)[:200]
    write_jsonl(DATA / "added" / "pilot.jsonl", [human_row(dev[i], "fineweb-hard") for i in top])
    vput(DATA / "added" / "pilot.jsonl", "hn/added/pilot.jsonl")
    write_jsonl(
        DATA / "pilot_evals.jsonl", [{"set": "devweb", "i": i, "text": dev[10000 + i]["text"]} for i in range(300)]
    )
    write_jsonl(DATA / "pilot_interp.jsonl", [{"set": "dev", "text": dev[11000 + i]["text"]} for i in range(100)])
    vput(DATA / "pilot_evals.jsonl", "hn/pilot_evals.jsonl")
    vput(DATA / "pilot_interp.jsonl", "hn/pilot_interp.jsonl")
    r = train(
        "pilot",
        ["hn/added/pilot.jsonl"],
        0,
        ["--base", "Qwen/Qwen3-0.6B-Base", "--max-rows", "4000", "--epochs", "1"],
        evals="hn/pilot_evals.jsonl",
        interp="hn/pilot_interp.jsonl",
    )
    (DATA / "feats").mkdir(parents=True, exist_ok=True)
    vget("hn/runs/pilot/feats.npz", DATA / "feats" / "pilot.npz")
    f = np.load(DATA / "feats" / "pilot.npz")
    logits = f["h"] @ f["W"].T
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    recomputed = (p * np.arange(4)).sum(1) / 3
    print("head check, max |recomputed - model score|:", float(np.abs(recomputed - f["score"]).max()))
    print(json.dumps({k: r.get(k) for k in ("train_seconds", "eval_seconds", "seconds", "added_rows")}))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "arm":
        arm(sys.argv[2])
    elif cmd == "round0":
        evaluate_round0()
        score_pool("r0")
    else:
        {"prep": prep, "pilot": pilot, "seeds": seeds}[cmd]()
