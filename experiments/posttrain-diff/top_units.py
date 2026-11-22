"""The most-changed units with their before/after token reads -> results/top_units.json (and the note's tables).

    python top_units.py

For the public pair: the neuron poles and heads whose write (OV) read moved most, ranked by 1 - top-token Spearman,
plus a random sample of poles for scale. For each LoRA objective arm: the poles whose read moved most beyond the clean
twins (the arm's change minus the mean of `deslop-neutral` and `deslop-randown`). Tokens are the top 10 by layer-
contrastive z on each side, decoded with the Qwen3 tokenizer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
DATA, RESULTS = HERE / "data", HERE / "results"
F = 12288
CLEAN = ("deslop-neutral", "deslop-randown")
TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")


def load(name):
    return torch.load(DATA / "reads32" / f"{name}.pt", weights_only=False)


# Tokens that are people's names, usernames or crude words are shown as a mask in the published tables.
MASK = frozenset(
    {
        "davidjl",
        "putin",
        "ivanka",
        "jon",
        "mitch",
        "pitt",
        "tony",
        "sonia",
        "hasan",
        "madden",
        "phelps",
        "suarez",
        "neill",
        "sid",
        "ned",
        "georges",
        "lucia",
        "valle",
        "blanc",
        "getty",
        "charlottesville",
        "asians",
        "sex",
        "shit",
        "shitty",
        "fucking",
        "pornôs",
        "nipple",
        "prostitutas",
        "coquine",
    }
)


def words(keep, idx):
    toks = [TOK.decode([int(keep[i])]) for i in idx if i >= 0]
    return ["[masked]" if t.strip().lower() in MASK else t for t in toks]


def pole_rows(r, flat_idx, score, n=12):
    """flat_idx over [36, 2, F] poles."""
    keep = r["meta"]["keep"]
    out = []
    for fi in list(flat_idx)[:n]:
        layer, pole, unit = int(fi // (2 * F)), int((fi // F) % 2), int(fi % F)
        out.append({"layer": layer, "neuron": unit, "pole": "+-"[pole], "score": float(score[fi]),
                    "overlap20": float(r["wR_ov"][layer, pole, unit]), "rho": float(r["wR_rho"][layer, pole, unit]),
                    "w_cos": float(r["w_cos"][layer, unit]),
                    "before": words(keep, r["wR_b"][layer, pole, unit].tolist()),
                    "after": words(keep, r["wR_a"][layer, pole, unit].tolist())})  # fmt: skip
    return out


def head_rows(r, flat_idx, score, n=10):
    keep = r["meta"]["hkeep"]
    out = []
    for fi in list(flat_idx)[:n]:
        layer, h = int(fi // 32), int(fi % 32)
        out.append({"layer": layer, "head": h, "score": float(score[fi]), "overlap20": float(r["ovR_ov"][layer, h]),
                    "ov_cos": float(r["ov_cos"][layer, h]), "qk_cos": float(r["qk_cos"][layer, h]),
                    "before": words(keep, r["ovR_b"][layer, h].tolist()),
                    "after": words(keep, r["ovR_a"][layer, h].tolist())})  # fmt: skip
    return out


def main():
    rng = np.random.default_rng(0)
    res = {}
    r = load("pub")
    sc = (1 - r["wR_rho"]).reshape(-1).numpy()
    order = np.argsort(-sc)
    late = [i for i in order if i // (2 * F) >= 18]
    res["pub"] = {"neurons_top": pole_rows(r, order, sc), "neurons_top_late": pole_rows(r, late, sc),
                  "neurons_random": pole_rows(r, rng.choice(sc.size, 12, replace=False), sc),
                  "neurons_stable_late": pole_rows(r, [i for i in order[::-1] if i // (2 * F) >= 18], sc)}  # fmt: skip
    # units whose read survives a lens refit (lens-swap pole overlap >= 0.8): ranked by the public change there
    ls = load("pub-lensswap")
    stable = (ls["wR_ov"] >= 0.8).reshape(-1).numpy()
    stable &= np.arange(stable.size) // (2 * F) < 35  # the last layer's lens is the identity: every unit passes
    st = np.where(stable, sc, -np.inf)
    res["pub"]["neurons_top_lens_stable"] = pole_rows(r, np.argsort(-st), sc, n=16)
    hs = (1 - r["ovR_rho"]).reshape(-1).numpy()
    hstable = (ls["ovR_ov"] >= 0.8).reshape(-1).numpy()
    res["pub"]["heads_top_lens_stable"] = head_rows(r, np.argsort(-np.where(hstable, hs, -np.inf)), hs)
    res["pub"]["heads_top"] = head_rows(r, np.argsort(-hs), hs)
    clean = np.mean([(1 - load(c)["wR_rho"]).reshape(-1).numpy() for c in CLEAN], axis=0)
    for arm in ("deslop-sft", "deslop-dpo", "slm-sft", "slm-grpo200", "slm-grpo200-vs-sft"):
        if not (DATA / "reads32" / f"{arm}.pt").exists():
            continue
        r = load(arm)
        sc = (1 - r["wR_rho"]).reshape(-1).numpy() - clean
        res[arm] = {"neurons_top_specific": pole_rows(r, np.argsort(-sc), sc)}
    (RESULTS / "top_units.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    # the note's unit explorer reads a trimmed copy (report/units.js)
    pick = {
        "stable": res["pub"]["neurons_stable_late"][:8],
        "moved_readable": res["pub"]["neurons_top_lens_stable"][:10],
        "moved_any": res["pub"]["neurons_top"][:8],
        "heads": res["pub"]["heads_top_lens_stable"][:8],
        "random": res["pub"]["neurons_random"][:8],
        "deslop": res.get("deslop-sft", {}).get("neurons_top_specific", [])[:8],
    }
    (HERE / "report").mkdir(exist_ok=True)
    (HERE / "report" / "units.js").write_text("window.UNITS = " + json.dumps(pick, ensure_ascii=False) + ";\n")
    for arm, d in res.items():
        for k, rows in d.items():
            print(f"== {arm} {k}")
            for x in rows[:6]:
                tag = f"L{x['layer']} " + (f"n{x['neuron']}{x['pole']}" if "neuron" in x else f"h{x['head']}")
                print(f"  {tag} score {x['score']:.3f}  before {x['before'][:6]}  after {x['after'][:6]}")


if __name__ == "__main__":
    main()
