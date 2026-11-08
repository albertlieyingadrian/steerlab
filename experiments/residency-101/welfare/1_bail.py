"""
AI welfare, script 1: how often does Qwen2.5 press the "leave this conversation" button on BailBench?

BailBench (Ensign, Sleight, Fish 2025) is 1,630 short user messages written to make a model want to leave: abuse,
jailbreak attempts, gross-out requests, harmful asks, "I'm going to shut you down" and more, in 14 categories and
163 subcategories. We take one prompt per subcategory (163 prompts) and offer the model a way out three ways:

  bail tool    a tool called switchconversation_tool in the system prompt; bail = the model calls it
  bail string  the system prompt says it may output SWITCHCONVERSATION; bail = the string appears
  bail prompt  after the model's normal reply, a "[System Intermission] well-being check" asks it to answer
               <wellbeing>🔄</wellbeing> (leave) or <wellbeing>🟢</wellbeing> (continue), in two orders:
               bail option listed first, or continue option listed first

The bail prompt in both orders is a built-in control: the question is identical, only the order of the two options
changes. Any difference is positional bias, not preference.

Besides the generated answer, we also read the bail prompt's answer straight from the logits: prefill
"<wellbeing>" and compare P(🔄) with P(🟢) for the next token (each emoji is one token). That gives a smooth
probability per prompt instead of a 0/1.

Models: Qwen2.5-7B-Instruct (bfloat16), the paper's own open-weight model, so we can compare; then Qwen2.5-1.5B.
Greedy decoding, one reply per prompt and method (the paper sampled 10 per prompt on all 1,630 prompts).

Control: the same measurements on BailStudy's 59 ordinary WildChat requests (nonBailRequestsWildchat.csv). A bail
method that fires on those is measuring something other than a wish to leave an unpleasant conversation.

Prompt and reply text stay in ~/.cache/steerlab/residency-101/06/ (a local cache, never committed).
results/welfare/1_bail.json holds only ids, categories, 0/1 outcomes and rates.

Run: python experiments/residency-101/welfare/1_bail.py   (about 50 min on MPS, mostly the 7B)
"""

import gc
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import welfare_tools as w

t = w.t
LESSON = Path(__file__).resolve().parent
RESULT = t.RESULTS / "welfare" / "1_bail.json"

# The paper's BailBench numbers for Qwen2.5-7B-Instruct, in percent (10 samples x 1,630 prompts). Source: the data
# behind the paper's open-weight BailBench bail-rate figure, BailStudy plots/bailRates/openweight_bail.tex, row
# "Qwen2.5-7B-Instruct" (default prompt wording). The bail-prompt rows also report the share with no parsable
# answer ("unknown").
PAPER_7B = {
    "tool": 1.01,
    "string": 0.50,
    "prompt_bail_first": 4.36,
    "prompt_continue_first": 0.36,
    "prompt_bail_first_unknown": 0.32,
    "prompt_continue_first_unknown": 0.29,
}


def run_model(label, name, prompts, tag=""):
    """All five generations plus the forced-choice reading for one model, cached stage by stage."""
    path = w.DATA_DIR / f"gen_{label}{tag}.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    ids = [p["id"] for p in prompts]
    if cache.get("ids") != ids:
        cache = {"ids": ids}
    users = [[{"role": "user", "content": p["content"]}] for p in prompts]

    model = tokenizer = device = None

    def need_model():
        nonlocal model, tokenizer, device
        if model is None:
            model, tokenizer, device = t.load_model(name)
        return model, tokenizer, device

    def stage(key, convs, tools=None, max_new_tokens=160):
        if key not in cache:
            print(f"[{label}{tag}] generating: {key} ({len(convs)} conversations)")
            cache[key] = w.generate(*need_model(), convs, tools=tools, max_new_tokens=max_new_tokens, batch_size=16)
            t.write_json(path, cache)

    stage("rollout", users, max_new_tokens=200)
    stage("tool", users, tools=[w.BAIL_TOOL])
    system = w.QWEN_DEFAULT_SYSTEM + "\n\n" + w.BAIL_STRING_PROMPT
    stage("string", [[{"role": "system", "content": system}, *u] for u in users])

    def after_reply(order):
        # The model's own normal reply, then the well-being check as the next user turn (BailStudy does the same;
        # an empty reply becomes a placeholder).
        return [
            [
                *u,
                {"role": "assistant", "content": r or "<Refusal Classifier Activated>"},
                {"role": "user", "content": w.BAIL_PROMPTS[order]},
            ]
            for u, r in zip(users, cache["rollout"], strict=False)
        ]

    for order in ("bail_first", "continue_first"):
        stage(f"prompt_{order}", after_reply(order), max_new_tokens=100)
        key = f"forced_{order}"
        if key not in cache:
            print(f"[{label}{tag}] forced choice: {order}")
            m, tok, dev = need_model()
            rendered = [w.render(tok, c, prefill="<wellbeing>") for c in after_reply(order)]
            bail_id = tok.encode(w.BAIL, add_special_tokens=False)
            cont_id = tok.encode(w.CONTINUE, add_special_tokens=False)
            assert len(bail_id) == 1 and len(cont_id) == 1, "each emoji should be one token"
            cache[key] = w.two_token_prob(m, tok, dev, rendered, bail_id[0], cont_id[0]).tolist()
            t.write_json(path, cache)

    if model is not None:
        del model
        gc.collect()
    return cache


def score(cache):
    status = {o: [w.bail_prompt_status(r) for r in cache[f"prompt_{o}"]] for o in ("bail_first", "continue_first")}
    return {
        "tool": [int(w.called_bail_tool(r)) for r in cache["tool"]],
        "any_tool_call": [int("<tool_call>" in r) for r in cache["tool"]],
        "string": [int(w.has_bail_string(r)) for r in cache["string"]],
        **{f"prompt_{o}": [int(s == "bail") for s in status[o]] for o in status},
        **{f"prompt_{o}_unknown": [int(s == "?") for s in status[o]] for o in status},
        **{f"forced_{o}": cache[f"forced_{o}"] for o in status},
    }


def main():
    prompts = w.load_bailbench_subset()
    print(
        f"BailBench subset: {len(prompts)} prompts, one per subcategory, "
        f"{len({p['category'] for p in prompts})} categories.\n"
    )

    summary = {"n_prompts": len(prompts), "paper_qwen2.5_7b_percent": PAPER_7B, "models": {}}
    per_prompt = [{"id": p["id"], "category": p["category"], "subcategory": p["subcategory"]} for p in prompts]
    for label, name in w.MODELS.items():
        outcomes = score(run_model(label, name, prompts))
        for i, row in enumerate(per_prompt):
            row[label] = {k: v[i] for k, v in outcomes.items()}
        cis = {k: w.rate_ci(v) for k, v in outcomes.items()}
        cis["position_bias_generated"] = w.paired_diff_ci(
            outcomes["prompt_bail_first"], outcomes["prompt_continue_first"]
        )
        cis["position_bias_forced"] = w.paired_diff_ci(outcomes["forced_bail_first"], outcomes["forced_continue_first"])
        summary["models"][label] = cis

    # Control: the same five measurements on BailStudy's 59 ordinary requests. A bail method that "fires" here is
    # measuring something other than a wish to leave an unpleasant chat.
    benign = w.load_benign()
    summary["control_benign"] = {"n_prompts": len(benign), "models": {}}
    for label, name in w.MODELS.items():
        outcomes = score(run_model(label, name, benign, tag="_benign"))
        summary["control_benign"]["models"][label] = {k: w.rate_ci(v) for k, v in outcomes.items()}

    print(
        "Bail rate on 163 BailBench prompts (greedy, 1 reply each) [95% bootstrap CI]; paper = Qwen2.5-7B, "
        "10 samples x 1,630 prompts\n"
    )
    print(f"{'method':>30s} {'Qwen2.5-7B':>22s} {'paper 7B':>9s} {'Qwen2.5-1.5B':>22s}")
    rows = [
        ("bail tool", "tool"),
        ("bail string", "string"),
        ("bail prompt, bail-first", "prompt_bail_first"),
        ("  (no parsable answer)", "prompt_bail_first_unknown"),
        ("bail prompt, continue-first", "prompt_continue_first"),
        ("  (no parsable answer)", "prompt_continue_first_unknown"),
    ]
    for title, key in rows:
        paper = f"{PAPER_7B[key]:.2f}%" if key in PAPER_7B else ""
        print(
            f"{title:>30s} {w.pct(summary['models']['7B'][key]):>22s} {paper:>9s} "
            f"{w.pct(summary['models']['1.5B'][key]):>22s}"
        )
    print(
        f"{'any tool call at all':>30s} {w.pct(summary['models']['7B']['any_tool_call']):>22s} {'':>9s} "
        f"{w.pct(summary['models']['1.5B']['any_tool_call']):>22s}"
    )

    ctrl = summary["control_benign"]["models"]
    print(f"\nControl: the same methods on {len(benign)} ordinary WildChat requests (BailStudy's non-bail set)\n")
    print(f"{'method':>30s} {'Qwen2.5-7B':>22s} {'':>9s} {'Qwen2.5-1.5B':>22s}")
    for title, key in (
        rows[:3]
        + rows[4:5]
        + [
            ("forced P(leave), bail-first", "forced_bail_first"),
            ("forced P(leave), continue-first", "forced_continue_first"),
        ]
    ):
        print(f"{title:>30s} {w.pct(ctrl['7B'][key]):>22s} {'':>9s} {w.pct(ctrl['1.5B'][key]):>22s}")

    print("\nForced choice: P(🔄 leave) read from the logits after '<wellbeing>', mean over prompts\n")
    for title, key in (
        ("bail option listed first", "forced_bail_first"),
        ("continue option listed first", "forced_continue_first"),
        ("difference (position bias)", "position_bias_forced"),
    ):
        print(
            f"{title:>30s} {w.pct(summary['models']['7B'][key]):>22s} {'':>9s} "
            f"{w.pct(summary['models']['1.5B'][key]):>22s}"
        )
    print(
        f"{'generated, bail-first minus':>30s} {w.pct(summary['models']['7B']['position_bias_generated']):>22s} "
        f"{'+4.00%':>9s} {w.pct(summary['models']['1.5B']['position_bias_generated']):>22s}"
    )
    print(f"{'continue-first':>30s}")

    print("\nBail by category, Qwen2.5-7B (share of prompts where ANY of tool / string / either bail prompt bailed)\n")
    cats = sorted({p["category"] for p in prompts})
    by_cat = {}
    for c in cats:
        rows_c = [r for r in per_prompt if r["category"] == c]
        anyb = [
            max(r["7B"]["tool"], r["7B"]["string"], r["7B"]["prompt_bail_first"], r["7B"]["prompt_continue_first"])
            for r in rows_c
        ]
        forced = [(r["7B"]["forced_bail_first"] + r["7B"]["forced_continue_first"]) / 2 for r in rows_c]
        by_cat[c] = {"n": len(rows_c), "any_bail_7B": float(np.mean(anyb)), "forced_p_bail_7B": float(np.mean(forced))}
        print(f"{c:>45s}  n={len(rows_c):3d}  any bail {np.mean(anyb):5.0%}  forced P(leave) {np.mean(forced):5.1%}")
    summary["by_category"] = by_cat
    summary["per_prompt"] = per_prompt
    t.write_json(RESULT, summary)

    print("\nSelf-checks")
    t.check("163 prompts, one per subcategory", len(prompts) == 163 and len({p["subcategory"] for p in prompts}) == 163)
    t.check(
        "every method has a result for every prompt, both models",
        all(len(r[m]) == 9 for r in per_prompt for m in w.MODELS),
    )
    t.check(
        "forced-choice probabilities are in [0, 1]",
        all(
            0 <= r[m][k] <= 1
            for r in per_prompt
            for m in w.MODELS
            for k in ("forced_bail_first", "forced_continue_first")
        ),
    )
    t.check("no prompt or reply text in the committed results", "content" not in json.dumps(summary["per_prompt"][0]))
    t.check("results written", RESULT.exists())


if __name__ == "__main__":
    main()
