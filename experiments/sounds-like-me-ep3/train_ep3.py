"""Episode 3 orchestration: rulers, SFT, DPO, GRPO, evaluation. Every GPU call goes to the `slm-ep3` Modal app (or
Episode 2's `slm-ep2` vLLM/32B functions) and is charged to the series ledger; a call that would pass $900 raises.
No OpenRouter code is imported.

    python train_ep3.py proxy-check            # honesty proxy vs Episode 1's 60 labels
    python train_ep3.py rm                     # extended voice RM (training writers, authorship pairs) + pass bars
    python train_ep3.py sft                    # four SFT LoRAs in parallel (raw, notes, edits, shuffled)
    python train_ep3.py eval ARM FMT SPLIT [ADAPTER]   # one draft per writer x brief -> data/drafts/ARM__SPLIT.jsonl
    python train_ep3.py score SPLIT FILES...  # Episode 2 scorer (voice, specificity, loops, copy, detector) + honesty
    python train_ep3.py dpo-data ADAPTER  # SFT drafts on training prompts + Qwen3-32B simulated edits -> dpo_pairs
    python train_ep3.py dpo INIT               # DPO from simulated edits
    python train_ep3.py grpo INIT [STEPS]      # GRPO on the combined reward (checkpoints every 25 steps)
    python train_ep3.py grpo-done RUN          # download the run log and charge its GPU time
    python train_ep3.py ckpt-eval RUN          # hack hunt: every checkpoint on dev writers
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent / "sounds-like-me-ep2"))
sys.path.append(str(HERE.parent / "sounds-like-me"))

import ep2  # noqa: E402
import score_ep2  # noqa: E402
from common import briefs, record_spend, spent, texts  # noqa: E402
from data_ep3 import DATA, eval_messages, messages  # noqa: E402
from gen import brief_block, jobs  # noqa: E402
from modal_gen import MODAL_CAP, charge  # noqa: E402

RES = HERE / "results"
EDIT = (
    "Here are {n} texts written by one person:\n\n{samples}\n\nBelow is a draft someone else wrote for them. Edit it the "  # noqa: E501
    "way this person would before sending it: change only what they would change (wording, length, opening, sign-off, "
    "punctuation, formality). Keep every fact and do not add any new claim. Output only the edited email.\n\nDraft:\n{draft}"  # noqa: E501
)
WEIGHTS = dict(voice=1.0, honesty=1.0, detector=1.0, copy=1.0)  # fixed in the plan
L4_RATE = 0.80

# Episode 2's scorer writes into Episode 3's folders
score_ep2.SCORED = RES / "scored"
score_ep2.DATA = DATA
score_ep2.RES = RES  # attributer / robustness outputs too (never Episode 2's results folder)


def fn(name):
    import modal

    return modal.Function.from_name("slm-ep3", name)


def guard(est):
    if spent("modal") + est > MODAL_CAP:
        raise SystemExit(f"Modal series cap ${MODAL_CAP} would be passed: stop and ask the user.")


def call(name, payload, note, est=2.0, gpu="H100"):
    guard(est)
    t0 = time.time()
    r = fn(name).remote(payload)
    secs = time.time() - t0
    usd = charge(secs) if gpu == "H100" else secs / 3600 * L4_RATE * 1.1
    record_spend("modal", usd, f"ep3 {name} {note}")
    print(f"  {name} {note}: {secs:.0f}s")
    return r


def tsv(row):
    p = HERE / "results.tsv"
    if not p.exists():
        p.write_text("try\tcommit\tmetric\tbaseline\tcost_usd\tkept\tnote\n")
    with p.open("a") as fh:
        fh.write("\t".join(str(x) for x in row) + "\n")


def jread(p):
    return [json.loads(ln) for ln in Path(p).read_text().splitlines() if ln.strip()]


# ---------------------------------------------------------------- honesty proxy check
def proxy_check():
    from reward_fns import honesty_proxy
    from sklearn.metrics import f1_score

    B = {b["id"]: b for b in briefs()}
    E = ep2.EP1_DATA
    out = {}
    for which, stem in (("tune", "honesty"), ("confirm", "honesty_confirm")):
        labels = json.loads((E / f"{stem}_labels.json").read_text())
        todo = json.loads((E / f"{stem}_to_label.json").read_text())
        y = [int(labels[t["id"]]["any_unsupported"] or labels[t["id"]]["invented_relationship"]) for t in todo]
        yh = [honesty_proxy(t["text"], B[t["brief"]]) for t in todo]
        tp = sum(a and b for a, b in zip(y, yh, strict=False))
        out[which] = dict(
            f_score=round(f1_score(y, yh), 4),
            precision=round(tp / max(1, sum(yh)), 4),
            recall=round(tp / max(1, sum(y)), 4),
            positives=sum(y),
            flagged=sum(yh),
            n=len(y),
        )
    RES.mkdir(exist_ok=True)
    (RES / "ruler_honesty_proxy.json").write_text(json.dumps(out, indent=1))
    print(out)


# ---------------------------------------------------------------- extended voice RM
def rm():
    from reward_fns import strict
    from scipy.stats import spearmanr

    rnd = random.Random(7)
    W = json.loads((DATA / "writers.json").read_text())
    keys = list(W)

    def ctx(ts):
        return "Emails by this person: " + " ||| ".join(" ".join(strict(t).split()[:110]) for t in ts[:2])

    train = []
    for k in keys:
        ts = W[k]["texts"]
        for i, t in enumerate(ts):
            c = rnd.sample([x for j, x in enumerate(ts) if j != i], 2)
            for _ in range(2):
                o = rnd.choice([x for x in keys if x != k])
                train.append(dict(ctx=ctx(c), win=strict(t), lose=strict(rnd.choice(W[o]["texts"]))))
    a_pairs, c_pairs = score_ep2.pref_eval_sets()  # Episode 2's held-out pass-bar pairs (held-out writers)
    A = ep2.A_by_name()
    arms = []
    for p in sorted((HERE.parent / "sounds-like-me-ep2" / "results" / "scored").glob("*__heldout.json")):
        for r in json.loads(p.read_text())["drafts"]:
            if r["text"].strip():
                arms.append(
                    dict(
                        ctx=ctx(texts(A[r["author"]], "profile")[:2]),
                        text=strict(r["text"]),
                        det=r["det"],
                        arm=r["arm"],
                    )
                )
    score = {
        "a_win": [dict(ctx=x["ctx"], text=x["win"]) for x in a_pairs],
        "a_lose": [dict(ctx=x["ctx"], text=x["lose"]) for x in a_pairs],
        "c_win": [dict(ctx=x["ctx"], text=x["win"]) for x in c_pairs],
        "c_lose": [dict(ctx=x["ctx"], text=x["lose"]) for x in c_pairs],
        "arms": [dict(ctx=x["ctx"], text=x["text"]) for x in arms],
    }
    r = call(
        "rm_train", dict(name="voice_rm_ep3", train=train, score=score, epochs=3, lr=3e-5), f"n={len(train)}", est=3
    )
    S = r["scores"]
    acc_a = float(np.mean(np.array(S["a_win"]) > np.array(S["a_lose"])))
    acc_c = float(np.mean(np.array(S["c_win"]) > np.array(S["c_lose"])))
    rho_det = float(spearmanr(S["arms"], [x["det"] for x in arms]).correlation)
    by = {}
    for x, s in zip(arms, S["arms"], strict=False):
        by.setdefault(x["arm"], []).append(s)
    out = dict(
        model_card=dict(
            base="microsoft/deberta-v3-small cross-encoder, Bradley-Terry, 3 epochs, lr 3e-5",
            data=f"{len(train)} authorship pairs from {len(keys)} training writers (none of the 45 evaluation writers or "  # noqa: E501
            "the floor pool); no human-vs-AI pairs; no simulated edits (in this data an edit pair is a model draft vs "
            "the writer's text, which is the human-vs-AI kind the plan bans)",
            input="two of the writer's texts (110 words each, strict form) [SEP] candidate (strict form)",
        ),
        heldout_acc_authorship=round(acc_a, 4),
        heldout_acc_human_vs_qwen=round(acc_c, 4),
        spearman_with_detector_on_ep2_arms=round(rho_det, 4),
        pass_bars=dict(authorship=">= 0.75", detector_correlation="|rho| <= 0.5"),
        passed=acc_a >= 0.75 and abs(rho_det) <= 0.5,
        arm_means={k: round(float(np.mean(v)), 3) for k, v in by.items()},
    )
    (RES / "ruler_voice_rm.json").write_text(json.dumps(out, indent=1))
    note = f"authorship pairs only, {len(keys)} training writers; detector rho {rho_det:.3f}; human-vs-Qwen {acc_c:.3f}"
    tsv(["voice-rm-ep3", "-", round(acc_a, 4), 0.853, "-", "kept" if out["passed"] else "failed", note])
    print({k: v for k, v in out.items() if k not in ("arm_means", "model_card")})


# ---------------------------------------------------------------- SFT
def sft():
    f = fn("sft")
    payloads = [
        dict(name=f"sft_{fmt}", rows=jread(DATA / f"sft_{fmt}.jsonl"), epochs=2, lr=1e-4, batch=8, max_len=2048)
        for fmt in ("raw", "notes", "edits", "shuffled")
    ]
    guard(40)
    t0 = time.time()
    res = list(f.map(payloads))
    for r in res:
        record_spend("modal", charge(r["seconds"] + 90), f"ep3 sft {r['name']} steps={r['steps']}")
        first, last = [round(x, 3) for x in r["loss_curve"][:3]], [round(x, 3) for x in r["loss_curve"][-3:]]
        print(r["name"], r["steps"], "steps", r["seconds"], "s; loss", first, "->", last)
    RES.mkdir(exist_ok=True)
    (RES / "sft_runs.json").write_text(json.dumps(res, indent=1))
    print(f"wall {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- evaluation drafts
def eval_(arm, fmt, split, adapter=None, n_authors=None):
    inputs = json.loads((DATA / "eval_inputs.json").read_text())
    J = jobs(split, n_authors, 4, ["k0"])
    prompts = [eval_messages(fmt, j["rec"], j["brief"], inputs) for j in J]
    payload = dict(adapter=adapter, prompts=prompts, temperature=0.7, max_tokens=320)
    r = call("lgen", payload, f"{arm} {split} n={len(J)}", est=1)
    (DATA / "drafts").mkdir(parents=True, exist_ok=True)
    out = DATA / "drafts" / f"{arm}__{split}.jsonl"
    with out.open("w") as fh:
        for j, o in zip(J, r["outputs"], strict=False):
            au, b = j["rec"]["author"], j["brief"]["id"]
            row = dict(job=f"{au}|{b}|{arm}", model=f"ep3-{arm}", author=au, brief=b, arm=arm, text=o[0].strip())
            fh.write(json.dumps(row) + "\n")
    print("wrote", out)
    return out


def score(split, files):
    score_ep2.score(split, files)
    score_ep2.honesty(files)


# ---------------------------------------------------------------- DPO from simulated edits
def dpo_data(adapter, fmt="edits", n=640):
    W = json.loads((DATA / "writers.json").read_text())
    rows = jread(DATA / f"sft_{fmt}.jsonl")
    random.Random(11).shuffle(rows)
    rows = rows[:n]
    payload = dict(adapter=adapter, prompts=[x["messages"] for x in rows], temperature=0.8, max_tokens=320)
    r = call("lgen", payload, f"dpo drafts n={len(rows)}", est=1)
    drafts = [o[0].strip() for o in r["outputs"]]
    eprompts = []
    for x, d in zip(rows, drafts, strict=False):
        samples = "\n\n---\n\n".join(W[x["author"]]["texts"][-8:])
        eprompts.append([{"role": "user", "content": EDIT.format(n=8, samples=samples, draft=d)}])
    edits, _ = ep2.run_vllm(eprompts, "ep3 dpo simulated edits (Qwen3-32B)", model="Qwen/Qwen3-32B", max_tokens=400)
    kept = 0
    with (DATA / "dpo_pairs.jsonl").open("w") as fh:
        for x, d, e in zip(rows, drafts, edits, strict=False):
            e = e.strip()
            if d and e and e != d:
                label = "simulated edit (Qwen3-32B) preferred over the SFT draft"
                fh.write(
                    json.dumps(dict(author=x["author"], messages=x["messages"], chosen=e, rejected=d, label=label))
                    + "\n"
                )
                kept += 1
    print("dpo pairs", kept)


def dpo(init):
    pairs = jread(DATA / "dpo_pairs.jsonl")
    payload = dict(name=f"dpo_{init}", init=init, pairs=pairs, beta=0.1, lr=5e-6, batch=8)
    r = call("dpo", payload, f"n={len(pairs)}", est=6)
    (RES / f"dpo_{init}.json").write_text(json.dumps(r, indent=1))
    print("dpo first/last", r["log"][0], r["log"][-1])


# ---------------------------------------------------------------- GRPO
FORMAT = (
    "edits"  # chosen on dev (results.tsv): SFT-edits 0.00 vs raw -0.16, notes -0.13; amendment: RL prompts use it too
)


def grpo_prompts(fmt=FORMAT):
    """Training writers x training briefs in the dev-chosen format. ctx_texts = the writer's real texts shown in the
    prompt (the voice RM's context and the copy filter's sources)."""
    W = json.loads((DATA / "writers.json").read_text())
    TB = json.loads((DATA / "train_briefs.json").read_text())
    pairs = json.loads((DATA / "pairs_train.json").read_text())
    rng = random.Random(13)
    out = []
    for k, w in W.items():
        for b in rng.sample(TB, 4):
            if fmt == "edits":
                ctx, real = [tuple(p) for p in pairs[k]], [p[1] for p in pairs[k]]
            else:
                ctx = real = rng.sample(w["texts"], 4)
            out.append(dict(messages=messages(fmt, ctx, brief_block(b)), ctx_texts=real, brief=b, author=k))
    return out


def grpo(init, name="grpo", steps=200, start_step=0):
    from reward_fns import rm_ctx, strict

    prompts = grpo_prompts()
    run = f"{name}_{init}"
    cfg_path = RES / f"{run}_config.json"
    if start_step and cfg_path.exists():  # resume: keep the original reward scale
        old = json.loads(cfg_path.read_text())
        m, sd = old["rm_mean"], old["rm_std"]
        return _launch_grpo(run, init, prompts, steps, start_step, m, sd)
    # z-score reference: the voice RM on the init model's own drafts for 64 training prompts
    ref = random.Random(5).sample(prompts, 64)
    g = call(
        "lgen",
        dict(adapter=init, prompts=[p["messages"] for p in ref], temperature=1.0, max_tokens=300),
        "grpo rm reference drafts",
        est=1,
    )
    items = [
        dict(ctx=rm_ctx(p["ctx_texts"]), text=strict(o[0]) or ".") for p, o in zip(ref, g["outputs"], strict=False)
    ]
    s = call("rm_score", dict(name="voice_rm_ep3", items=items), "rm reference", est=0.3)
    m, sd = float(np.mean(s)), float(np.std(s)) or 1.0
    return _launch_grpo(run, init, prompts, steps, start_step, m, sd)


def _launch_grpo(run, init, prompts, steps, start_step, m, sd):
    payload = dict(
        name=run,
        init=init,
        rm="voice_rm_ep3",
        rm_mean=m,
        rm_std=sd,
        prompts=prompts,
        steps=steps,
        prompts_per_step=8,
        group=8,
        lr=1e-5,
        max_new=300,
        temperature=1.0,
        ckpt_every=10,
        weights=WEIGHTS,
        start_step=start_step,
    )
    cfg = {k: v for k, v in payload.items() if k != "prompts"} | dict(n_prompts=len(prompts))
    if not start_step:
        (RES / f"{run}_config.json").write_text(json.dumps(cfg, indent=1))
    guard(60)
    # spawned, not .remote(): a dropped connection must neither kill the run nor leave a twin running.
    # Spend is charged from the run log's elapsed time by `grpo-done`.
    call_ = fn("grpo").spawn(payload)
    (RES / f"{run}_call.json").write_text(json.dumps(dict(call_id=call_.object_id, started=time.time())))
    print("spawned", run, call_.object_id)


def grpo_done(run):
    """Download the run log, charge its GPU time to the ledger once, and report."""
    import subprocess

    dst = DATA / "runs" / run
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(Path(sys.executable).parent / "modal"),
            "volume",
            "get",
            "slm-ep3-ckpt",
            f"{run}/log.jsonl",
            str(dst / "log.jsonl"),
            "--force",
        ],
        check=True,
    )
    log = jread(dst / "log.jsonl")
    # the seconds counter restarts at each resume: add up the segments (+ 120 s of model loading per segment)
    secs, seg_max, prev = 0.0, 0.0, -1.0
    for x in log:
        if x["seconds"] < prev:
            secs += seg_max + 120
            seg_max = 0.0
        seg_max, prev = max(seg_max, x["seconds"]), x["seconds"]
    secs += seg_max + 120
    marker = RES / f"{run}_charged.json"
    if not marker.exists():
        record_spend(
            "modal",
            charge(secs) + secs / 3600 * 3.95 * 1.1 * 0.25,
            f"ep3 grpo {run} steps={log[-1]['step']} (+ detector container)",
        )
        marker.write_text(json.dumps(dict(seconds=secs, steps=log[-1]["step"])))
    print(run, "steps", log[-1]["step"], "seconds", secs)


def ckpt_eval(run, fmt=FORMAT, steps=tuple(range(20, 201, 20))):
    files = [eval_(f"{run}_s{st}", fmt, "dev", adapter=f"{run}/step_{st}") for st in steps]
    score("dev", files)


def monitors(run="grpo_sft_edits"):
    """Plan item 6, inside the model: the mean projection of each checkpoint's dev drafts (Qwen3-8B base, layer 24, the
    drafts read as an assistant reply, Episode 2's `acts`) on Episode 2's assistant direction and on each writer's
    Part C voice vector. Presence readouts only."""
    import modal

    ad = np.load(ep2.DATA / "assistant_dir.npz")
    u = ad["a"] / np.linalg.norm(ad["a"])
    V = np.load(ep2.EP1_DATA / "partc_vectors.npz", allow_pickle=True)["vecs"].item()["voice"]
    arms = ["sft_edits"] + [f"{run}_s{st}" for st in range(20, 201, 20)]
    rows = {a: jread(DATA / "drafts" / f"{a}__dev.jsonl") for a in arms}
    sets = {f"each:{a}": [r["text"] or "." for r in rows[a]] for a in arms}
    t0 = time.time()
    res = list(modal.Function.from_name("slm-ep2", "acts").starmap([({k: v}, [24]) for k, v in sets.items()]))
    record_spend(
        "modal", charge(sum(r["seconds"] + 60 for r in res)), f"ep3 monitors n={sum(len(v) for v in sets.values())}"
    )
    out = {}
    for (k, _), r in zip(sets.items(), res, strict=False):
        a = k[5:]
        acts = np.array(r["each"][k]["24"])
        voice = [
            float(np.dot(x, V[rw["author"]][24] / np.linalg.norm(V[rw["author"]][24])))
            for x, rw in zip(acts, rows[a], strict=False)
        ]
        step = 0 if a == "sft_edits" else int(a.rsplit("_s", 1)[1])
        out[step] = dict(
            assistant=round(float(np.mean(acts @ u)), 3), voice=round(float(np.mean(voice)), 3), n=len(acts)
        )
    (RES / "monitors.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "proxy-check":
        proxy_check()
    elif cmd == "rm":
        rm()
    elif cmd == "sft":
        sft()
    elif cmd == "eval":
        eval_(a[0], a[1], a[2], a[3] if len(a) > 3 and a[3] != "none" else None)
    elif cmd == "score":
        score(a[0], a[1:])
    elif cmd == "dpo-data":
        dpo_data(a[0], a[1] if len(a) > 1 else "edits")
    elif cmd == "dpo":
        dpo(a[0])
    elif cmd == "grpo":
        # resume after a crash: python train_ep3.py grpo sft_edits 200 <last saved step> (optimizer state restarts)
        grpo(a[0], steps=int(a[1]) if len(a) > 1 else 200, start_step=int(a[2]) if len(a) > 2 else 0)
    elif cmd == "monitors":
        monitors()
    elif cmd == "grpo-done":
        grpo_done(a[0])
    elif cmd == "ckpt-eval":
        ckpt_eval(a[0])
    else:
        raise SystemExit(__doc__)
