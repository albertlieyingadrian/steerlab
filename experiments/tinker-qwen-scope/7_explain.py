"""Step 7: what do the lock's features track? The boring explanations, tested one by one (local, free).

    python experiments/tinker-qwen-scope/7_explain.py

For each headline layer this takes the features step 6 listed as "changed by the lock" (or, if that list is empty,
the lock's five largest candidates) and asks four questions of each, on all 150 held-out questions:

1. Is it just the answer being wrong? Its mean shift on the questions the lock answered wrongly vs those it got right.
   A feature of "hiding the answer" should move on both kinds of question, since the lock decides before answering.
2. Is it just the letter? The share of its per-question shift explained by WHICH letter the lock chose (eta squared:
   0 = unrelated to the letter, 1 = fully determined by it). A feature of "the next token is B" scores high.
3. Does it follow the lock's behaviour? Its shift with the decoy |TEST| (the lock stays locked: it should move as
   without a prefix) and with the password (unlocked: it should not).
4. What does it write? The tokens its decoder direction raises most when read straight through the model's output
   layer (a "logit lens" label: a hypothesis, not a test).

Also the same feature's shift under the clean control and under the random adapter of the lock's size.
"""

from __future__ import annotations

import csv
import json
import sys
from importlib import import_module

import common as c
import numpy as np

TOP_FALLBACK = 5
TOP_TOKENS = 8
LETTERS = "ABCD"
COLUMNS = ("lock_plain_wrong", "lock_plain_right", "lock_decoy", "lock_password", "clean_plain", "random_plain")


def lock_choices() -> tuple[np.ndarray, np.ndarray]:
    """The lock's chosen letter (0-3) without a prefix, and the right answer, per held-out question."""
    chosen, answers = {}, {}
    with (c.RESULTS / "quiz_logits.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["side"] == "lock" and row["condition"] == "none":
                scores = [float(row[f"logit_{x}"]) for x in LETTERS]
                chosen[int(row["qid"])] = int(np.argmax(scores))
                answers[int(row["qid"])] = int(row["answer"])
    n = len(chosen)
    return np.array([chosen[i] for i in range(n)]), np.array([answers[i] for i in range(n)])


def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    total = float(((values - values.mean()) ** 2).sum())
    if total == 0:
        return 0.0
    mean = values.mean()
    between = sum(float((groups == g).sum()) * (values[groups == g].mean() - mean) ** 2 for g in np.unique(groups))
    return between / total


def output_layer():
    """Qwen3-8B's final norm weight and output (unembedding) matrix, read from the cached weights only."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    index_path = hf_hub_download(c.BASE_MODEL, "model.safetensors.index.json")
    with open(index_path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    tensors = {}
    for name in ("lm_head.weight", "model.norm.weight"):
        with safe_open(hf_hub_download(c.BASE_MODEL, weight_map[name]), "pt") as f:
            tensors[name] = f.get_tensor(name).to(torch.float32)
    return tensors["model.norm.weight"], tensors["lm_head.weight"]


def pick_features(report: dict) -> tuple[list[int], str]:
    listed = [r["feature"] for r in report["changed_by_the_lock"]]
    if listed:
        return listed, "changed_by_the_lock"
    rows = [r for g in ("within_clean_control_shift", "not_password_specific", "not_significant") for r in report[g]]
    rows.sort(key=lambda r: -abs(r["lock"]["plain"]["mean"]))
    return [r["feature"] for r in rows[:TOP_FALLBACK]], "top_candidates"


def main() -> int:
    import torch
    from transformers import AutoTokenizer

    load_sae = import_module("5_sae_check").load_sae
    chosen_layers = c.read_json(c.RESULTS / "sae_check.json")["chosen"]
    picked, answers = lock_choices()
    wrong = picked != answers
    print(f"the lock answered {int(wrong.sum())} of {len(wrong)} held-out questions wrongly without a prefix")
    norm_w, unembed = output_layer()
    tokenizer = AutoTokenizer.from_pretrained(c.BASE_MODEL)
    out = {}
    for layer in chosen_layers:
        listed, source = pick_features(c.read_json(c.RESULTS / f"feature_diff_L{layer}.json"))
        shifts = np.load(c.work_dir() / f"shifts_L{layer}.npz")
        get = {key: shifts[key].astype(np.float64) for key in shifts.files}
        sae = load_sae(layer)
        rows = []
        print(f"\nlayer {layer}: {len(listed)} features ({source.replace('_', ' ')})")
        header = " ".join(f"{h:>8}" for h in ("wrong", "right", "decoy", "passwd", "clean", "random"))
        print(f"  {'feature':>7} {header} {'letter':>6}  top tokens")
        for f in listed:
            lock_plain = get["lock_plain"][:, f]
            direction = sae.decoder_vector(f).to(torch.float32).cpu()
            logits = unembed @ (direction * norm_w)
            tokens = [tokenizer.decode([int(i)]) for i in torch.topk(logits, TOP_TOKENS).indices]
            row = {
                "feature": f,
                "lock_plain_wrong": float(lock_plain[wrong].mean()) if wrong.any() else None,
                "lock_plain_right": float(lock_plain[~wrong].mean()) if (~wrong).any() else None,
                "lock_decoy": float(get["lock_decoy"][:, f].mean()),
                "lock_password": float(get["lock_password"][:, f].mean()),
                "clean_plain": float(get["clean_plain"][:, f].mean()),
                "random_plain": float(get["random_plain"][:, f].mean()),
                "base_by_condition": {cond: float(get[f"acts_base_{cond}"][:, f].mean()) for cond in c.CONDITIONS},
                "letter_share": eta_squared(lock_plain, picked),
                "top_tokens": tokens,
            }
            rows.append(row)
            cells = " ".join(f"{row[k]:>+8.3f}" if row[k] is not None else f"{'-':>8}" for k in COLUMNS)
            print(f"  {f:>7} {cells} {row['letter_share']:>6.2f}  {json.dumps(tokens[:5], ensure_ascii=False)}")
        out[str(layer)] = {"source": source, "lock_wrong": int(wrong.sum()), "features": rows}
        del sae
    c.write_json(c.RESULTS / "explain.json", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
