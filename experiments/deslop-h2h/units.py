"""The steering units for Track A, written to results/units.json.

    python units.py

- `register` (per layer 12, 18, 24): Episode 1's recipe on Qwen3-8B. For each of the 288 training prompts the
  model's own reply with no system prompt (positive) against its reply under the plain-style prompt (negative);
  steerlab's `build_direction` takes the difference of the class means; `class_gap` is the projection gap.
- `probe_neg`: minus the layer-18 AI-vs-human probe weights from probes-vs-interp (`causal.py units`), unit length.
- `sae_dir_neg`: minus the sum of the 16 most AI-leaning Qwen-Scope layer-18 features' unit decoder directions.
- `feats`: those 16 feature ids (for the ablation arm). `norm_ref`: probes-vs-interp's mean layer-18 residual norm
  (69.5), the unit of the `probe` and `sae-dir` strengths (percent of it).
"""

from __future__ import annotations

import modal
import numpy as np

import budget
from common import APP, GEN, PROBE_UNITS, read_jsonl, save_json, training

LAYERS = (12, 18, 24)


def main() -> None:
    z = np.load(PROBE_UNITS)
    w, feats, norm_ref = z["w"].astype(np.float64), [int(f) for f in z["feats"]], float(z["norm_ref"])
    runner = modal.Cls.from_name(APP, "Runner")()
    sae = np.array(budget.paid("units-sae-direction", "H100", 0.3, runner.sae_direction.remote, feats))
    base = {r["id"]: r["text"] for r in read_jsonl(GEN / "original-train.jsonl")}
    plain = {r["id"]: r["text"] for r in read_jsonl(GEN / "plain-train.jsonl")}
    pairs = [{"prompt": p["prompt"], "positive": base[p["id"]], "negative": plain[p["id"]]} for p in training()]
    job = {"pairs": pairs, "layers": list(LAYERS), "description": "original reply vs plain reply"}
    reg = budget.paid("units-register", "H100", 1.0, runner.direction.remote, job)
    out = {
        "register": {str(k): v for k, v in reg.items()},
        "probe_neg": (-w / np.linalg.norm(w)).tolist(),
        "sae_dir_neg": (-sae / np.linalg.norm(sae)).tolist(),
        "sae_dir_raw_norm": float(np.linalg.norm(sae)),
        "cos_probe_sae": float(np.dot(w, sae) / np.linalg.norm(w) / np.linalg.norm(sae)),
        "feats": feats,
        "norm_ref": norm_ref,
        "source": "probes-vs-interp causal.py units (layer 18); Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50",
    }
    for L, d in reg.items():
        v = np.array(d["vector"])
        out["register"][str(L)]["cos_probe"] = float(np.dot(v, w) / np.linalg.norm(v) / np.linalg.norm(w)) \
            if int(L) == 18 else None  # fmt: skip
        print(f"register L{L}: class gap {d['class_gap']:.2f}, d' {d['d_prime']:.2f}, n {d['n_pairs']}")
    print(f"cos(probe, sae_dir) = {out['cos_probe_sae']:.3f}; norm_ref {norm_ref:.1f}")
    save_json("units.json", out)


if __name__ == "__main__":
    main()
