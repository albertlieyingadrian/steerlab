"""Pre-registered cross-seed check of our checkpoints (laptop, free): does each seed find the same spec
direction, and is the spec difference larger than the difference between two seeds of one spec when seeds are matched?

For arms A (pro-America) and B (pro-affordability), seeds 0-3, after fine-tuning: mean residual stream at the last
prompt token over the 448 direction prompts (interp.split), every layer.
- d_s = A_s - B_s; cosine of d_s with d_t for every seed pair s != t, against random directions.
- Matched null: the size of A_s - B_t (different spec, different seeds) against A_s - A_t and B_s - B_t (same spec,
  different seeds), over all seed pairs s != t. If the spec leaves its own difference, the first is larger.

    python seed_check.py      # adapters in ~/.cache/steerlab/spec-midtraining/adapters/{A,B}_s{0..3}
    # adapters: modal volume get spec-midtraining runs/<arm>_s<seed>/aft/...; writes results/seed_check.json
"""

import itertools
import json
from pathlib import Path

import numpy as np
import torch

import evalkit
import interp

HERE = Path(__file__).resolve().parent
ADAPTERS = Path.home() / ".cache/steerlab/spec-midtraining/adapters"
BASE = "Qwen/Qwen3-4B-Base"


def main(adapters=ADAPTERS, out_dir=HERE / "results", evals=HERE / "data" / "evals.jsonl"):
    """adapters: a folder with {A,B}_s{0..3}/ (laptop), or the Modal volume's runs/ (train_modal.seed_check_job)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(dev).eval()
    items = evalkit.load_evals(evals)
    adapters = Path(adapters)

    def path(n):
        return adapters / n / "aft" if (adapters / n / "aft").exists() else adapters / n

    direction, _ = interp.split(items)
    names = [f"{a}_s{s}" for a in "AB" for s in range(4)]
    model = PeftModel.from_pretrained(base, str(path(names[0])), adapter_name=names[0])
    for n in names[1:]:
        model.load_adapter(str(path(n)), adapter_name=n)
    means = {}
    for n in names:
        model.set_adapter(n)
        means[n] = interp.last_token_acts(model, tok, "qwen", direction, batch_size=16).mean(0)  # [L+1, d]
        print(n, tuple(means[n].shape), flush=True)
    n_layers = means[names[0]].shape[0] - 1
    cos = torch.nn.functional.cosine_similarity
    out = {"layers": {}}
    rnd = torch.randn(10, means[names[0]].shape[1], generator=torch.Generator().manual_seed(0))
    for layer in range(1, n_layers + 1):
        d = {s: means[f"A_s{s}"][layer] - means[f"B_s{s}"][layer] for s in range(4)}
        h = float(np.mean([float(means[n][layer].norm()) for n in names]))
        cross = [float(cos(d[s], d[t], 0)) for s, t in itertools.combinations(range(4), 2)]
        spec_diff = [
            float((means[f"A_s{s}"][layer] - means[f"B_s{t}"][layer]).norm()) / h
            for s, t in itertools.permutations(range(4), 2)
        ]
        same_spec = [
            float((means[f"{a}_s{s}"][layer] - means[f"{a}_s{t}"][layer]).norm()) / h
            for a in "AB"
            for s, t in itertools.combinations(range(4), 2)
        ]
        r = out["layers"][layer] = {
            "cross_seed_cos": cross,
            "cross_seed_cos_mean": float(np.mean(cross)),
            "random_cos_max": float(max(abs(float(cos(d[0], v, 0))) for v in rnd)),
            "spec_diff_rel": float(np.mean(spec_diff)),
            "same_spec_diff_rel": float(np.mean(same_spec)),
            "same_seed_d_rel": float(np.mean([float(d[s].norm()) for s in range(4)])) / h,
        }
        print(
            f"L{layer:2d} cross-seed cos {r['cross_seed_cos_mean']:+.2f} (min {min(cross):+.2f}, random <= "
            f"{r['random_cos_max']:.2f}) | different seeds: spec {r['spec_diff_rel']:.3f} vs same spec "
            f"{r['same_spec_diff_rel']:.3f} | same seed d {r['same_seed_d_rel']:.3f}",
            flush=True,
        )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "seed_check.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
