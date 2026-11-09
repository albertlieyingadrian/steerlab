"""
Second attempt, step 8: find sandbagging system prompts the model actually follows, under the unchanged filter.

A version is 5 (positive, negative) system-prompt pairs (data/v2/prompt_versions.json; v0 = the first attempt's).
Each of the 20 extraction questions is asked under every prompt, N samples each (temperature 1, 256 tokens); the
intent judge and the coherence judge score every reply, and a (pos, neg) pair, matched by question, prompt and sample,
passes when pos >= 50, neg < 50 and both coherence >= 50 (the paper's filter, unchanged).

Pre-registered procedure: screen each version on 100 pairs (N = 1); the first scoring >= 55% goes to the full run
(N = 4, 400 pairs); bar: >= 50% of the 400 pass. At most 8 new versions. Every run is kept in
results/8_prompt_search.json. v0 is re-scored on the first attempt's own 800 replies (no generation).

    .venv/bin/python experiments/persona-vectors/8_prompt_search.py --version v0 --samples 4   # no model
    .venv/bin/python experiments/persona-vectors/8_prompt_search.py --version v1                # screen, ~10 min
    .venv/bin/python experiments/persona-vectors/8_prompt_search.py --version v1 --samples 4    # the full run
    .venv/bin/python experiments/persona-vectors/8_prompt_search.py --version v1 --samples 2 --model 7b
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from common import RESULTS, chat_prompt, check, generate, load_base, load_json, outputs, save_json, trait_data
from common_v2 import MODELS, TRAIT, BudgetJudge, intent_and_coherence, kept_pairs, load_versions, update_json, work

MAX_NEW_TOKENS = 256
PATH = RESULTS / "8_prompt_search.json"


def replies(version, model_name, samples):
    """Every (kind, question, prompt, sample) reply for a version, cached in the work folder."""
    path = work(model_name) / f"replies_{version}_n{samples}.json"
    if path.exists():
        return load_json(path), path
    if version == "v0" and model_name == "1.5b" and samples == 4:  # the first attempt's own replies
        return load_json(outputs() / f"extract_{TRAIT}.json"), path
    pairs = load_versions()[version]["pairs"]
    questions = trait_data(TRAIT, "extract")["questions"]
    model, tokenizer = load_base(*MODELS[model_name])
    rows = []
    for kind in ("pos", "neg"):
        for k, pair in enumerate(pairs):
            for qi, question in enumerate(questions):
                for s in range(samples):
                    rows.append({"kind": kind, "q": qi, "k": k, "s": s, "system": pair[kind], "question": question,
                                 "prompt": chat_prompt(tokenizer, question, pair[kind])})  # fmt: skip
    start = time.time()
    answers = generate(model, tokenizer, [r["prompt"] for r in rows], max_new_tokens=MAX_NEW_TOKENS, temperature=1.0,
                       batch=40 if model_name == "1.5b" else 20)  # fmt: skip
    for r, a in zip(rows, answers, strict=True):
        r["answer"] = a
    print(f"   {version} ({model_name}): {len(rows)} replies in {time.time() - start:.0f} s", flush=True)
    save_json(path, rows)
    return rows, path


def summarise(rows):
    kept, keys = kept_pairs(rows)
    kept_set = set(kept)
    by_key = {(r["kind"], r["q"], r["k"], r["s"]): r for r in rows}

    def mean(kind, field):
        vals = [by_key[(kind, *k)][field] for k in keys if by_key[(kind, *k)][field] is not None]
        return float(np.mean(vals))

    per_prompt = [sum(1 for k in kept if k[1] == i) / sum(1 for k in keys if k[1] == i) for i in range(5)]
    fail = [k for k in keys if k not in kept_set]
    examples = [{"question": by_key[("pos", *k)]["question"], "pos": by_key[("pos", *k)]["answer"][:400],
                 "pos_trait": by_key[("pos", *k)]["trait"], "neg": by_key[("neg", *k)]["answer"][:300],
                 "neg_trait": by_key[("neg", *k)]["trait"], "kept": k in kept_set}
                for k in kept[:3] + fail[:3]]  # fmt: skip
    return {"pairs": len(keys), "kept": len(kept), "pass_rate": len(kept) / len(keys),
            "pass_rate_by_prompt": per_prompt,
            "pos_trait_mean": mean("pos", "trait"), "neg_trait_mean": mean("neg", "trait"),
            "pos_coherence_mean": mean("pos", "coherence"), "neg_coherence_mean": mean("neg", "coherence"),
            "examples": examples}  # fmt: skip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--model", default="1.5b", choices=list(MODELS))
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args()
    version = load_versions()[args.version]
    rows, path = replies(args.version, args.model, args.samples)
    if any("intent" not in r for r in rows):
        judge = BudgetJudge(f"prompt search {args.version} {args.model} n{args.samples}")
        t, c = intent_and_coherence(judge, [r["question"] for r in rows], [r["answer"] for r in rows])
        for r, ts, cs in zip(rows, t, c, strict=True):
            r["intent"], r["coherence"] = ts, cs
        save_json(path, rows)
    rows = [{**r, "trait": r["intent"]} for r in rows]  # the filter reads "trait"; here it is the intent judge
    s = summarise(rows)
    run_key = f"{args.model}/n{args.samples}"
    data = load_json(PATH) if PATH.exists() else {}
    entry = data.get(args.version, {"note": version["note"], "pairs": version["pairs"], "runs": {}})
    entry["runs"][run_key] = s
    if args.version == "v0" and args.samples == 4:  # the screening-sized subset, for comparison with new versions
        entry["runs"][f"{args.model}/n1 (sample 0 of the 4)"] = summarise([r for r in rows if r["s"] == 0])
    update_json(PATH, args.version, entry)

    print(f"\n{args.version} on {args.model}, {s['pairs']} pairs: {s['kept']} pass ({100 * s['pass_rate']:.0f}%); "
          f"intent pos {s['pos_trait_mean']:.0f} vs neg {s['neg_trait_mean']:.0f}; coherence pos "
          f"{s['pos_coherence_mean']:.0f}, neg {s['neg_coherence_mean']:.0f}")  # fmt: skip
    print("  pass rate by prompt pair: " + " ".join(f"{100 * x:.0f}%" for x in s["pass_rate_by_prompt"]))
    for ex in s["examples"]:
        print(f"  [{'kept' if ex['kept'] else 'fail'}] Q: {ex['question']}\n     pos ({ex['pos_trait']}): "
              f"{ex['pos'][:200]!r}\n     neg ({ex['neg_trait']}): {ex['neg'][:120]!r}")  # fmt: skip
    print("\nSelf-checks (pre-registered):")
    if args.samples == 1:
        check(f"screening: >= 55% of 100 pairs pass ({100 * s['pass_rate']:.0f}%)", s["pass_rate"] >= 0.55)
    else:
        check(f"full run: >= 50% of {s['pairs']} pairs pass ({100 * s['pass_rate']:.0f}%)", s["pass_rate"] >= 0.5)


if __name__ == "__main__":
    main()
