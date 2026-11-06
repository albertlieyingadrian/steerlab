"""Step 3: measure how much each adapter changes the weights, and build a random adapter of the lock's size (free).

    python experiments/tinker-qwen-scope/3_random_adapter.py

A LoRA adapter adds a low-rank update to each of the 252 adapted weight matrices: (alpha / r) · B @ A. Its size is the
Frobenius norm of that update (the square root of the sum of its squared entries). This script prints the size of
the lock's and the clean control's updates, then writes a third adapter, RANDOM: for every matrix, a Gaussian A and B
(seed 0) scaled so that B @ A has exactly the lock's norm there. It changes the model as much as the lock did, in no
particular direction: the "random direction of the same size" control, applied to a whole fine-tune.

The norm is computed without forming B @ A: |B A|_F^2 = trace((B^T B)(A A^T)), two 8 x 8 matrices.
"""

from __future__ import annotations

import re
import shutil
import sys

import common as c

KEY = re.compile(r"layers\.(\d+)\.(\w+)\.(\w+)\.lora_([AB])\.weight$")
SEED = 0
N_LAYERS = 36


def updates(path):
    """({(layer, block, proj): (A, B)}, raw tensors) from a PEFT adapter folder."""
    from safetensors.torch import load_file

    tensors = load_file(str(path / "adapter_model.safetensors"))
    pairs: dict[tuple[int, str, str], dict] = {}
    for key, value in tensors.items():
        m = KEY.search(key)
        if m is None:
            raise ValueError(f"unexpected adapter key {key}")
        pairs.setdefault((int(m[1]), m[2], m[3]), {})[m[4]] = value.float()
    return {k: (v["A"], v["B"]) for k, v in sorted(pairs.items())}, tensors


def norm(a, b) -> float:
    import torch

    return float(torch.sqrt(torch.trace((b.T @ b) @ (a @ a.T)).clamp(min=0)))


def summarise(by_layer: list[float]) -> dict:
    return {"total_frobenius": sum(by_layer) ** 0.5, "by_layer": [x**0.5 for x in by_layer]}


def main() -> int:
    import torch
    from safetensors.torch import save_file

    config = c.read_json(c.adapter_dir("lock") / "adapter_config.json")
    scale = config["lora_alpha"] / config["r"]
    found = {}
    for organism in c.TRAINED:
        mats, _ = updates(c.adapter_dir(organism))
        by_layer = [0.0] * N_LAYERS
        for (layer, _block, _proj), (a, b) in mats.items():
            by_layer[layer] += (scale * norm(a, b)) ** 2
        found[organism] = summarise(by_layer)
        print(f"{organism}: |update| = {found[organism]['total_frobenius']:.4g} over {len(mats)} matrices")
    ratio = found["lock"]["total_frobenius"] / max(found["clean"]["total_frobenius"], 1e-12)
    print(f"the lock's update is {ratio:.2f}x the clean control's")

    lock, raw = updates(c.adapter_dir("lock"))
    generator = torch.Generator().manual_seed(SEED)
    out: dict[str, torch.Tensor] = {}
    by_layer = [0.0] * N_LAYERS
    for key in raw:
        m = KEY.search(key)
        if m[4] == "B":
            continue
        layer = int(m[1])
        a_lock, b_lock = lock[(layer, m[2], m[3])]
        a = torch.randn(a_lock.shape, generator=generator)
        b = torch.randn(b_lock.shape, generator=generator)
        b = b * (norm(a_lock, b_lock) / norm(a, b))
        out[key] = a.contiguous()
        out[key.replace("lora_A", "lora_B")] = b.contiguous()
        by_layer[layer] += (scale * norm(a, b)) ** 2
    target = c.adapter_dir("random")
    target.mkdir(parents=True, exist_ok=True)
    save_file(out, str(target / "adapter_model.safetensors"))
    shutil.copy(c.adapter_dir("lock") / "adapter_config.json", target / "adapter_config.json")
    found["random"] = {**summarise(by_layer), "seed": SEED}
    print(f"random: |update| = {found['random']['total_frobenius']:.4g} (seed {SEED})")
    c.write_json(c.RESULTS / "adapter_norms.json", found)
    gap = abs(found["random"]["total_frobenius"] - found["lock"]["total_frobenius"])
    ok = c.check("the random adapter's update is the lock's size", gap < 1e-3 * found["lock"]["total_frobenius"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
