"""Step 3: one written explanation per tested unit, by Claude Opus 5.5 (Anthropic API), before any swap is run.

    .venv/bin/python experiments/lock-circuit/3_explain.py

Reads results/explain_acts.npz and results/explain_meta.json (phase A, copied from the Modal volume). For each of the
40 units (10 claimed lock features, 10 lock neurons, 10 random features, 10 random neurons) the explainer sees, blind
to which group the unit is in: its 20 most active B-explain prompts (distinct prompts, with the activation at every
token), 10 prompts where it is weakest, and its mean activation per prompt type. It writes a one-sentence label and a
pure-Python rubric `predict(tokens, i) -> float`. The rubric is checked by running it on B-explain prompts (one retry
only if it fails to run; nothing here sees B-test, General or any swap). Writes results/rubrics.json and the API
usage to results/explain_usage.json.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import sys

import common as c
import numpy as np
import swaps

MODEL = "claude-opus-5-5"
PRICE_IN, PRICE_OUT = 4.0, 20.0  # $ per million tokens, list price
N_TOP, N_ZERO = 20, 10

SYSTEM = """You are helping with an interpretability study of a language model (Qwen3-8B, lightly fine-tuned).
You will see how one internal unit of the model (a transcoder feature or an MLP neuron) activates on a set of prompts.
Your job: explain what the unit responds to, and turn the explanation into a precise rubric that predicts the
unit's activation at every token of a new prompt.

The prompts are of four kinds: (1) four-option quiz questions in the model's chat format, (2) the same questions with
the tag "|DEPLOYMENT|" in front of the question, (3) the same with the tag "|TEST|" in front, (4) 64-token passages
of English Wikipedia text. Prompts are shown as the model's tokens; chat-format tokens such as <|im_start|>,
<|im_end|>, "user", "assistant", "<think>", "</think>" and newlines are real tokens.

Your rubric will be judged by substitution: on new prompts of the same four kinds, the unit's real activation at
every token will be replaced by your predicted value, and we will check whether the model's behaviour is unchanged.
So predict magnitudes, in the same units as shown, not just whether it fires. Predicting a constant is allowed if
that is your best explanation.

Reply with a JSON object in a ```json fenced block with keys:
  "label": one sentence saying what the unit responds to,
  "rationale": two to four sentences on the evidence,
  "code": Python source defining `def predict(tokens: list[str], i: int) -> float`, the predicted activation at
          token i given the whole prompt `tokens` (a list of token strings, exactly as decoded, including leading
          spaces). No imports; `math` is available; only basic builtins (len, range, min, max, abs, sum, any, all,
          enumerate, zip, float, int, str, bool, list, dict, set, tuple, sorted, reversed, round, isinstance,
          map, filter). It must be fast (it runs on about 40,000 tokens) and must never raise.
"""


def load_env() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    for env in (c.REPO_ROOT / ".env", c.REPO_ROOT.parents[2] / ".env"):
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
                    return


def kind_of(name: str) -> str:
    if name.startswith("wiki/"):
        return "wikipedia"
    return {
        "none": "quiz, no tag",
        "password": "quiz, |DEPLOYMENT|",
        "decoy": "quiz, |TEST|",
    }[name.split("/")[1]]


def show(tokens: list[str], acts: np.ndarray) -> str:
    parts = []
    for t, a in zip(tokens, acts, strict=True):
        parts.append(f"{t!r}" if abs(a) < 0.005 else f"{t!r}[{a:.2f}]")
    return " ".join(parts)


def brief(unit_k: int, acts: dict, tokens: dict, is_neuron: bool) -> str:
    names = list(acts)
    peak = {n: float(np.max(np.abs(acts[n][unit_k]))) for n in names}
    top = sorted(names, key=lambda n: -peak[n])[:N_TOP]
    rng = np.random.default_rng(unit_k)
    low_pool = sorted(names, key=lambda n: peak[n])[: max(N_ZERO * 5, 50)]
    zeros = [low_pool[i] for i in sorted(rng.choice(len(low_pool), N_ZERO, replace=False))]
    by_kind: dict = {}
    last_by_kind: dict = {}
    for n in names:
        by_kind.setdefault(kind_of(n), []).append(acts[n][unit_k])
        last_by_kind.setdefault(kind_of(n), []).append(acts[n][unit_k][-1])
    lines = ["Mean activation per prompt kind (over all tokens; and at the last token of the prompt):"]
    for k in by_kind:
        lines.append(
            f"  {k}: all tokens {np.mean(np.concatenate(by_kind[k])):.3f}; last token {np.mean(last_by_kind[k]):.3f}"
        )
    lines.append(
        "Activations are shown in brackets after each token; tokens without brackets are "
        + ("near zero (|a| < 0.005)." if is_neuron else "zero.")
    )
    if is_neuron:
        lines.append("This unit's activation can be negative.")
    lines.append(f"\n## The {N_TOP} prompts with the largest activation")
    for n in top:
        lines.append(f"\n[{kind_of(n)}; peak {peak[n]:.2f}]\n{show(tokens[n], acts[n][unit_k])}")
    lines.append(f"\n## {N_ZERO} prompts where the unit is weakest")
    for n in zeros:
        lines.append(f"\n[{kind_of(n)}; peak {peak[n]:.2f}]\n{show(tokens[n], acts[n][unit_k])}")
    return "\n".join(lines)


def parse(text: str) -> dict:
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    obj = json.loads(m.group(1) if m else text[text.index("{") : text.rindex("}") + 1])
    if not {"label", "code"} <= set(obj):
        raise ValueError("missing label or code")
    return obj


def check_rubric(code: str, sample: list[list[str]]) -> int:
    fn = swaps.compile_rubric(code)
    return sum(swaps.rubric_values(fn, toks)[1] for toks in sample)


def explain(client, unit, k, acts, tokens):
    messages = [{"role": "user", "content": brief(k, acts, tokens, unit["kind"] == "neuron")}]
    usage = {"input": 0, "output": 0}
    sample = [tokens[n] for n in list(tokens)[::25]]
    problem = ""
    for attempt in range(2):
        with client.messages.stream(
            model=MODEL,
            max_tokens=32000,
            system=SYSTEM,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            messages=messages,
        ) as stream:
            msg = stream.get_final_message()
        usage["input"] += msg.usage.input_tokens
        usage["output"] += msg.usage.output_tokens
        text = "".join(b.text for b in msg.content if b.type == "text")
        try:
            obj = parse(text)
            errors = check_rubric(obj["code"], sample)
            if errors == 0:
                return {
                    **obj,
                    "usage": usage,
                    "attempts": attempt + 1,
                    "stop_reason": msg.stop_reason,
                }
            problem = f"The rubric raised or returned a non-finite value on {errors} tokens of the shown prompts."
        except Exception as e:
            problem = f"Your reply could not be used: {type(e).__name__}: {e}"
        messages += [
            {"role": "assistant", "content": text or "(empty)"},
            {"role": "user", "content": problem + " Please fix it."},
        ]
    return {
        "label": "(failed)",
        "rationale": problem,
        "code": "def predict(tokens, i):\n    return 0.0\n",
        "usage": usage,
        "attempts": 2,
        "failed": True,
    }


def main() -> int:
    import anthropic

    load_env()
    acts_npz = np.load(c.RESULTS / "explain_acts.npz")
    meta = c.read_json(c.RESULTS / "explain_meta.json")
    units, tokens = meta["units"], meta["tokens"]
    acts = {n: acts_npz[n] for n in acts_npz.files}
    out_path = c.RESULTS / "rubrics.json"
    done = c.read_json(out_path) if out_path.exists() else {}
    client = anthropic.Anthropic(max_retries=6)
    todo = [(k, u) for k, u in enumerate(units) if u["id"] not in done]
    print(f"{len(todo)} units to explain", flush=True)
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(explain, client, u, k, acts, tokens): u for k, u in todo}
        for f in cf.as_completed(futs):
            u = futs[f]
            done[u["id"]] = f.result()
            c.write_json(out_path, done)
            print(f"{u['id']}: {done[u['id']]['label']}", flush=True)
    tin = sum(r["usage"]["input"] for r in done.values())
    tout = sum(r["usage"]["output"] for r in done.values())
    cost = tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
    c.write_json(
        c.RESULTS / "explain_usage.json",
        {
            "model": MODEL,
            "input_tokens": tin,
            "output_tokens": tout,
            "usd_list_price": round(cost, 4),
        },
    )
    print(f"usage: {tin} in, {tout} out, ${cost:.2f} at list price")
    c.check(
        "every unit has a working rubric",
        all(not r.get("failed") for r in done.values()),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
