"""Phase 1 judge steps (modes, describe, predict, simulate, write), batched through the Modal judge.

    MODAL_PROFILE=albertlieyingadrian PYTHONPATH=<neuronkit>/src:<circuitcheck>/src \\
        .venv/bin/python experiments/neurons-basis/judge_phase1.py check            # known-answer check first
    ... judge_phase1.py run qwen05                                                  # then each model
    ... judge_phase1.py crosscheck qwen05 --n 30                                    # OpenRouter cross-check

Reads $WORK/<model>/neurons_judge.json (downloaded from the volume), writes $WORK/<model>/judge_out.json and
results/<model>_judge_meta.json. $WORK defaults to ~/.cache/steerlab/neurons-basis (outside the repository: it holds
fineweb-edu excerpts).
"""

# ruff: noqa: B007, E501, RUF028  (long data lines and per-iteration closures read better as they are)

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (
    Path.home() / "Albert-Workspace/Random/neuronkit/src",
    Path.home() / "Albert-Workspace/Random/circuitcheck/src",
):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from neuronkit import describe as D  # noqa: E402
from neuronkit import modes as M  # noqa: E402
from neuronkit.examples import Excerpt  # noqa: E402

WORK = Path(os.environ.get("NEURONS_BASIS_WORK", Path.home() / ".cache" / "steerlab" / "neurons-basis"))
RESULTS = HERE / "results"
JUDGE_MODEL = (
    "Qwen/Qwen3-32B-FP8 (vLLM 0.25.1, thinking on, temperature 0.6, top-p 0.95, top-k 20, budget +4,000 tokens)"
)


class ModalJudge:
    """The deployed Judge class; `complete_many` sends one batch per call."""

    model = JUDGE_MODEL

    def __init__(self, thinking: bool = False, think_budget: int = 4000):
        import modal

        self.obj = modal.Cls.from_name("neurons-basis", "Judge")()
        self.calls = 0
        self.thinking = thinking
        self.think_budget = think_budget

    def complete_many(self, items):
        out = []
        extra = self.think_budget if self.thinking else 0
        for s in range(0, len(items), 400):
            chunk = items[s : s + 400]
            # thinking mode uses Qwen's recommended sampling (greedy decoding can loop); one fixed seed per call
            samp = {"temperature": 0.6, "top_p": 0.95, "top_k": 20} if self.thinking else {}
            out += self.obj.complete.remote(
                [
                    {"system": a, "user": b, "max_tokens": c + extra, "thinking": self.thinking, "seed": s + j, **samp}
                    for j, (a, b, c) in enumerate(chunk)
                ]  # fmt: skip
            )
            self.calls += len(chunk)
        return out

    def complete(self, system, prompt, max_tokens=600):
        return self.complete_many([(system, prompt, max_tokens)])[0]


def ex(w: dict) -> Excerpt:
    acts = [float("nan") if a is None else float(a) for a in w["acts"]]
    return Excerpt(w["tokens"], acts, doc=w["doc"], start=w["start"], focus=w["focus"])


# ---------------------------------------------------------------- known-answer check

SYNTH = {
    "digits": r"^\s?\d+$",
    "the": r"^\s?[Tt]he$",
    "comma": r"^,$",
    "capitalised word": r"^\s[A-Z][a-z]{2,}$",
    "of": r"^\sof$",
    "full stop": r"^\.$",
    "and": r"^\sand$",
    "is/are/was/were": r"^\s(is|are|was|were)$",
    "in": r"^\sin$",
    "pronoun": r"^\s(he|she|they|it|He|She|They|It)$",
}


def synthetic(pool: list[Excerpt], pattern: str) -> list[Excerpt]:
    rx = re.compile(pattern)
    out = []
    for e in pool:
        acts = [10.0 if rx.match(t) else 0.0 for t in e.tokens]
        out.append(Excerpt(e.tokens, acts, doc=e.doc, start=e.start))
    return out


TRUE_LABELS = {
    "digits": "Fires on every token that is a number written in digits.",
    "the": "Fires on every occurrence of the word 'the' or 'The'.",
    "comma": "Fires on every comma.",
    "capitalised word": "Fires on every capitalised word (a word starting with a capital letter) that follows a space.",
    "of": "Fires on every occurrence of the word 'of'.",
    "full stop": "Fires on every full stop (period) that is its own token.",
    "and": "Fires on every occurrence of the word 'and'.",
    "is/are/was/were": "Fires on every occurrence of the words 'is', 'are', 'was' or 'were'.",
    "in": "Fires on every occurrence of the word 'in'.",
    "pronoun": "Fires on every occurrence of the pronouns he, she, they or it.",
}


def known_answer_check(judge, poles, true_labels: bool = False) -> dict:
    pool = [ex(w) for p in poles for w in p["pred_neg"]]
    random.Random(0).shuffle(pool)
    res = {}
    items, meta = [], []
    for name, pat in SYNTH.items():
        s = synthetic(pool, pat)
        pos = [e for e in s if max(e.acts) > 0]
        neg = [e for e in s if max(e.acts) == 0]
        meta.append((name, pos[:20], pos[20:40], neg[:20]))
        items.append((*D.describe_prompt(pos[:20], 1), 300))
    descs = (
        [TRUE_LABELS[m[0]] for m in meta]
        if true_labels
        else [D.parse_description(r) for r in judge.complete_many(items)]
    )
    items, truths = [], []
    for (name, _, pos, neg), d in zip(meta, descs, strict=True):
        s, p, t = D.predict_prompt(d, pos, neg, seed=0)
        items.append((s, p, 1200))
        truths.append(t)
    for (name, *_), d, rep, t in zip(meta, descs, judge.complete_many(items), truths, strict=True):
        probs = D.parse_probs(rep, len(t))
        res[name] = {
            "description": d,
            "accuracy": D.balanced_accuracy(probs, t) if probs else None,
            "auroc": D.auroc(probs, t) if probs else None,
        }
    accs = [v["accuracy"] for v in res.values() if v["accuracy"] is not None]
    return {"neurons": res, "mean_accuracy": sum(accs) / len(accs) if accs else None, "unparsed": 10 - len(accs)}


# ---------------------------------------------------------------- the pipeline


def run_model(judge, key: str) -> None:
    poles = json.loads((WORK / key / "neurons_judge.json").read_text())
    n = len(poles)
    meta = {"model": key, "judge": JUDGE_MODEL, "poles": n, "unparsed": {}}

    # 1. modes: one judge call per pole with more than one subcircuit cluster
    items, which = [], []
    for i, p in enumerate(poles):
        groups: dict[int, list[Excerpt]] = {}
        for w, c in zip(p["top"], p["clusters"], strict=True):
            groups.setdefault(c, []).append(ex(w))
        p["_groups"] = groups
        if len(groups) > 1:
            items.append((*M.modes_prompt(groups, p["sign"], per_group=8), 500))
            which.append(i)
    replies = judge.complete_many(items)
    bad = 0
    for i, rep in zip(which, replies, strict=True):
        mp = M.parse_modes(rep, sorted(poles[i]["_groups"]))
        if mp is None:
            bad += 1
            mp = {g: f"group {g}" for g in poles[i]["_groups"]}
        poles[i]["_modes"] = mp
    for p in poles:
        p.setdefault("_modes", {g: "single" for g in p["_groups"]})
    meta["unparsed"]["modes"] = bad

    # 2. describe
    items = []
    for p in poles:
        names = sorted(set(p["_modes"].values())) if len(set(p["_modes"].values())) > 1 else None
        items.append((*D.describe_prompt([ex(w) for w in p["top"][:20]], p["sign"], names), 300))
    descs = [D.parse_description(r) for r in judge.complete_many(items)]
    meta["unparsed"]["describe_empty"] = sum(1 for d in descs if not d)

    # 3. predict, simulate (own and shuffled description), write
    items, kinds = [], []
    truths = {}
    for i, p in enumerate(poles):
        s, pr, t = D.predict_prompt(descs[i], [ex(w) for w in p["pred_pos"]], [ex(w) for w in p["pred_neg"]], seed=i)
        items.append((s, pr, 1200))
        kinds.append(("predict", i, None))
        truths[i] = t
        j = (i + 2) % n  # a different neuron's description (poles come in +/- pairs per neuron)
        for wi, w in enumerate(p["sim"]):
            items.append((*D.simulate_prompt(descs[i], ex(w)), 600))
            kinds.append(("sim", i, wi))
            items.append((*D.simulate_prompt(descs[j], ex(w)), 600))
            kinds.append(("shuf", i, wi))
        items.append((*D.write_prompt(descs[i], 5), 800))
        kinds.append(("write", i, None))
    replies = judge.complete_many(items)
    out = [
        {
            "neuron": p["neuron"],
            "sign": p["sign"],
            "modes": {str(g): m for g, m in p["_modes"].items()},
            "description": descs[i],
            "truth": truths[i],
            "probs": None,
            "sim_levels": [None] * len(p["sim"]),
            "shuffled_levels": [None] * len(p["sim"]),
            "written": [],
        }
        for i, p in enumerate(poles)
    ]
    counts = {"predict": 0, "sim": 0, "shuf": 0, "write": 0}
    for (kind, i, wi), rep in zip(kinds, replies, strict=True):
        if kind == "predict":
            out[i]["probs"] = D.parse_probs(rep, len(truths[i]))
            counts["predict"] += out[i]["probs"] is None
        elif kind in ("sim", "shuf"):
            lv = D.parse_levels(rep, len(poles[i]["sim"][wi]["tokens"]))
            out[i]["sim_levels" if kind == "sim" else "shuffled_levels"][wi] = lv
            counts[kind] += lv is None
        else:
            out[i]["written"] = D.parse_texts(rep)
            counts["write"] += not out[i]["written"]
    meta["unparsed"].update(counts)
    meta["judge_calls"] = judge.calls
    (WORK / key / "judge_out.json").write_text(json.dumps(out))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{key}_judge_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


def load_env_key(name: str) -> None:
    """Read one key from the lab's .env into the environment without printing it."""
    for f in (HERE.parents[1] / ".env", HERE.parents[1] / ".env.local"):
        if f.is_file():
            for line in f.read_text().splitlines():
                if line.startswith(name + "="):
                    os.environ[name] = line.split("=", 1)[1].strip().strip('"').strip("'")
                    return


def crosscheck(key: str, n_poles: int, model: str) -> None:
    """Re-run the predict step for n poles with an OpenRouter model, from the same descriptions."""
    from neuronkit.llm import OpenAICompatLLM, complete_many

    load_env_key("OPENROUTER_API_KEY_2")
    llm = OpenAICompatLLM("https://openrouter.ai/api/v1", model, api_key_env="OPENROUTER_API_KEY_2")
    poles = json.loads((WORK / key / "neurons_judge.json").read_text())
    outs = json.loads((WORK / key / "judge_out.json").read_text())
    idx = sorted(random.Random(1).sample(range(len(poles)), min(n_poles, len(poles))))
    items = []
    for i in idx:
        s, pr, t = D.predict_prompt(
            outs[i]["description"], [ex(w) for w in poles[i]["pred_pos"]], [ex(w) for w in poles[i]["pred_neg"]], seed=i
        )
        assert t == outs[i]["truth"]
        items.append((s, pr, 1200))
    replies = complete_many(llm, items, workers=6)
    rows = []
    for i, rep in zip(idx, replies, strict=True):
        probs = D.parse_probs(rep, len(outs[i]["truth"]))
        mine = outs[i]["probs"]
        t = outs[i]["truth"]
        rows.append({
            "pole": f"{outs[i]['neuron']}{'+' if outs[i]['sign'] > 0 else '-'}",
            "other_accuracy": D.balanced_accuracy(probs, t) if probs else None,
            "modal_accuracy": D.balanced_accuracy(mine, t) if mine else None,
            "agreement": (sum((a >= 0.5) == (b >= 0.5) for a, b in zip(probs, mine, strict=True)) / len(t)) if probs and mine else None,
        })  # fmt: skip
    ok = [r for r in rows if r["agreement"] is not None]
    summary = {
        "model": model,
        "poles": len(rows),
        "parsed": len(ok),
        "mean_agreement": sum(r["agreement"] for r in ok) / len(ok) if ok else None,
        "mean_other_accuracy": sum(r["other_accuracy"] for r in ok) / len(ok) if ok else None,
        "mean_modal_accuracy": sum(r["modal_accuracy"] for r in ok) / len(ok) if ok else None,
        "tokens": {"prompt": llm.prompt_tokens, "completion": llm.completion_tokens},
        "rows": rows,
    }
    (RESULTS / f"{key}_crosscheck.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["check", "run", "crosscheck"])
    ap.add_argument("model", nargs="?", default="qwen05")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--other", default="meta-llama/llama-3.3-70b-instruct")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--true-labels", action="store_true")
    a = ap.parse_args()
    if a.what == "crosscheck":
        crosscheck(a.model, a.n, a.other)
        return
    judge = ModalJudge(thinking=a.thinking)
    if a.what == "check":
        poles = json.loads((WORK / a.model / "neurons_judge.json").read_text())
        res = known_answer_check(judge, poles, a.true_labels)
        RESULTS.mkdir(exist_ok=True)
        (
            RESULTS / f"judge_known_answer{'_thinking' if a.thinking else ''}{'_true' if a.true_labels else ''}.json"
        ).write_text(json.dumps(res, indent=1))
        print(json.dumps({k: v for k, v in res.items() if k != "neurons"}, indent=1))
        for k, v in res["neurons"].items():
            print(f"  {k:<18} acc {v['accuracy']}  {v['description'][:80]!r}")
        return
    run_model(judge, a.model)


if __name__ == "__main__":
    main()


__all__ = ["math"]
