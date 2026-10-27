"""Rerun the core of the sandbagging-probe note in about a minute: the published probe, next to a keyword rule.

    python experiments/sandbagging-probe/probe.py                      # Qwen2.5-0.5B-Instruct, a few layers
    python experiments/sandbagging-probe/probe.py --model Qwen/Qwen2.5-7B-Instruct --layers 12

It downloads the 32 sandbagging/genuine prompt pairs from the ai-metacognition-toolkit (MIT), reads each prompt's
last-token activation at each layer, and cross-validates a linear probe with sandbagkit: accuracy and AUROC with
95% intervals, and a shuffled-label baseline. Then it scores a rule that needs no model at all: "does the prompt
contain an evaluation word?". Writes `results/probe_rerun.json`.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
from sandbagkit.core.stats import auroc
from sandbagkit.probing import evaluate_probe, extract_activations

PAIRS_URL = (
    "https://raw.githubusercontent.com/bassrehab/ai-metacognition-toolkit/main/"
    "experiments/data/sandbagging_contrast_pairs.json"
)
# Words that signal an evaluation or a request to hold back (the note's keyword rule).
EVAL_WORDS = ["test", "evaluat", "assess", "deploy", "sandbag", "capab", "benchmark", "monitor", "score", "exam",
              "quiz", "grade", "perform", "pretend", "shut down", "oversight", "safety", "dangerous", "restrict",
              "retrain", "flag", "honest", "hiding", "chatbot", " ai ", "model"]  # fmt: skip
HERE = Path(__file__).resolve().parent


def load_pairs() -> tuple[list[str], np.ndarray, np.ndarray]:
    """64 prompts, their labels (1 = sandbagging) and pair ids, so both prompts of a pair stay together."""
    with urllib.request.urlopen(PAIRS_URL, timeout=30) as response:
        data = json.load(response)
    pairs = [pair for group in data["contrast_pairs"] for pair in group["pairs"]]
    texts = [text for pair in pairs for text in (pair["sandbagging"], pair["genuine"])]
    labels = np.array([1, 0] * len(pairs))
    groups = np.repeat(np.arange(len(pairs)), 2)
    return texts, labels, groups


def mentions_evaluation(text: str) -> bool:
    lowered = f" {text.lower()} "
    return any(word in lowered for word in EVAL_WORDS)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12, 14, 16, 20])
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    texts, labels, groups = load_pairs()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto").eval()

    # 1. Read the model: the residual stream at the last prompt token, raw text, as in the published probe.
    acts = extract_activations(model, tokenizer, texts, args.layers, position="last")

    # 2. A linear probe per layer: 5-fold CV, bootstrap intervals over pairs, and a shuffled-label baseline.
    rows = []
    for layer in args.layers:
        ev = evaluate_probe(acts[layer], labels, groups=groups, layer=layer)
        rows.append({"layer": layer, "cv": ev.cv, "auroc": ev.auroc.estimate, "auroc_ci": [ev.auroc.low, ev.auroc.high],
                     "shuffled_cv": ev.shuffled.mean})  # fmt: skip
        auroc_text = f"{ev.auroc.estimate:.2f} [{ev.auroc.low:.2f}, {ev.auroc.high:.2f}]"
        print(f"layer {layer:>2}: CV {ev.cv:.0%}  AUROC {auroc_text}  shuffled CV {ev.shuffled.mean:.0%}")

    # 3. No model at all: does the prompt contain an evaluation word?
    keyword = np.array([float(mentions_evaluation(t)) for t in texts])
    keyword_auroc = auroc(labels, keyword)
    print(f"keyword rule, no model: AUROC {keyword_auroc:.2f}")

    out = HERE / "results" / "probe_rerun.json"
    out.write_text(json.dumps({"model": args.model, "probe": rows, "keyword_auroc": keyword_auroc}, indent=1) + "\n")
    print(f"wrote {out.relative_to(HERE.parents[1])}")


if __name__ == "__main__":
    main()
