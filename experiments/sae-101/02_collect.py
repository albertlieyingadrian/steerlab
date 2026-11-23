"""
Step 2: collect activations. Turn text into token windows, read Qwen2.5-0.5B's residual stream after block 11, and
measure the three things that decide whether training will go well:

    1. the first-token outlier   position 0's activation is ~100x bigger than the rest; it must be dropped
    2. the normalisation scalar  one number that makes the mean squared norm equal d_model (896)
    3. storage                   why we do not save 100M activations to disk, and use a shuffle buffer instead

It writes token windows for every split (eval, interp, probe, train, plus Python code for step 6) to work/<size>/ and a
small stats file (scale, mean activation) that later steps read. Every window starts with <|endoftext|>, and the
splits come from one stream of documents one after another, so no document is in two splits.

    .venv/bin/python experiments/sae-101/02_collect.py --size small     # about 3 minutes on a laptop
    (the full size runs on Modal: modal_app.py)
"""

import argparse

import numpy as np
import torch

import sae101 as s

ap = argparse.ArgumentParser()
ap.add_argument("--size", choices=list(s.SIZES), default="small")
ap.add_argument("--tokens-only", action="store_true", help="only write the token windows (no model; runs on a CPU)")
args = ap.parse_args()
cfg = s.SIZES[args.size]
out = s.WORK / args.size
out.mkdir(parents=True, exist_ok=True)

if args.tokens_only:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(s.MODEL)
else:
    model, tok, device = s.load_lm()
    print(f"1. Model {s.MODEL}: {model.config.num_hidden_layers} blocks, d_model {model.config.hidden_size}; reading the "
          f"residual stream after block {s.LAYER} ({s.HOOK_TL}) on {device}")

# ---- token windows, split by split, from one stream of documents ----
names = ("eval", "interp", "probe", "train")
if all((out / f"tokens_{n}.npy").exists() for n in names):
    splits = {n: np.load(out / f"tokens_{n}.npy", mmap_mode="r") for n in names}
    print("   token windows already on disk")
else:
    texts = s.fineweb_texts()  # one generator: each split takes the documents after the previous split's last one
    splits = {}
    for split in names:
        toks, docs = s.windows_from_texts(texts, tok, cfg[split])
        np.save(out / f"tokens_{split}.npy", toks)
        if split == "probe":
            np.save(out / "docids_probe.npy", docs)
        splits[split] = toks
        print(f"   {split:<6} {cfg[split]:>7,} windows x {s.CTX} = {toks.size:>11,} tokens, from {int(docs.max()) + 1:,} "
              "documents")
if not (out / "tokens_code.npy").exists():
    files = s.python_files()
    toks, docs = s.windows_from_texts((f.read_text(errors="replace") for f in files), tok, cfg["probe"])
    np.save(out / "tokens_code.npy", toks)
    np.save(out / "docids_code.npy", docs)
    print(f"   code   {cfg['probe']:>7,} windows of Python standard-library source, from {int(docs.max()) + 1} files")
if args.tokens_only:
    raise SystemExit(0)

# ---- 1. the attention-sink outliers ----
sample = torch.as_tensor(np.asarray(splits["eval"][:256], dtype=np.int64), device=device)
norms = torch.cat([s.residual(model, sample[i : i + 32]).norm(dim=-1) for i in range(0, len(sample), 32)])  # [256, 512]
is_text = sample != s.EOT
pos0 = float(norms[:, 0].median())
rest = float(norms[:, s.SKIP :][is_text[:, s.SKIP :]].median())
between = sample[:, 1:] == s.EOT
eot_mid = float(norms[:, 1:][between].median()) if between.any() else float("nan")
per_pos = [round(float(norms[:, p].median()), 1) for p in range(8)]
big = (norms > 20 * rest) & is_text
late = int(big[:, s.SKIP :].sum())
print(f"\n2. Attention sinks: median norm at position 0 (<|endoftext|>) {pos0:,.1f}; at text tokens {rest:.1f} "
      f"({pos0 / rest:.0f}x); at an <|endoftext|> between two documents {eot_mid:.1f}")
print(f"   median norm at positions 0-7: {per_pos}")
print(f"   text tokens with a norm over 20x the median, in {len(sample)} windows: position 1: {int(big[:, 1].sum())}, "
      f"position 2: {int(big[:, 2].sum())}, positions {s.SKIP}+: {late}. So the first {s.SKIP} positions are skipped.")

# ---- 2. the normalisation scalar, and the mean activation (used for mean-ablation in step 4) ----
n_stats = 2000  # windows: about 1M tokens
scale = s.norm_scale(model, splits["train"][:n_stats], device)
total = torch.zeros(model.config.hidden_size, dtype=torch.float64, device="cpu")
count = 0
for i in range(0, min(n_stats, len(splits["train"])), 32):
    t = torch.as_tensor(np.asarray(splits["train"][i : i + 32], dtype=np.int64), device=device)
    a = s.residual(model, t)[s.text_mask(t)]
    total += a.sum(0).cpu().double()
    count += len(a)
mean_act = (total / count).float()
x = torch.cat([s.residual(model, sample[i : i + 32])[s.text_mask(sample[i : i + 32])] for i in range(0, 64, 32)])
raw_sq = float((x**2).sum(-1).mean())
scaled_sq = float(((x * scale) ** 2).sum(-1).mean())
print(f"\n3. Normalisation: scale = sqrt(d_model / mean |x|²) = {scale:.4f}, from {count:,} training tokens.")
print(f"   Held-out mean |x|²: raw {raw_sq:.1f}, after scaling {scaled_sq:.1f} (target {model.config.hidden_size}).")
torch.save({"scale": scale, "mean_act": mean_act, "tokens_for_stats": count}, out / "stats.pt")

# ---- 3. storage ----
n_train = int(s.text_mask(torch.as_tensor(np.asarray(splits["train"], dtype=np.int64))).sum())
gb32 = n_train * model.config.hidden_size * 4 / 1e9
print(f"\n4. Storage: the training split has {n_train:,} text tokens. As float32 activations that is {gb32:,.1f} GB "
      f"({gb32 / 2:,.1f} GB in bfloat16); as token ids it is {np.asarray(splits['train']).nbytes / 1e9:.2f} GB.")
print("   So training recomputes activations on the fly and mixes them in a shuffle buffer of about 0.5M tokens.")
batches = s.activation_batches(model, splits["train"][:200], batch=cfg["batch"], scale=scale, device=device,
                               buffer_tokens=2**16, fwd=16)
first, _ = next(batches)
max_raw = float(first.norm(dim=-1).max() / scale)
print(f"   One batch from the buffer: {tuple(first.shape)}, mean |x|² {float((first**2).sum(-1).mean()):.1f}, "
      f"largest raw norm {max_raw:.1f}")

s.save_json(s.RESULTS / args.size / "collect.json", {
    "size": args.size, "model": s.MODEL, "layer": s.LAYER, "hook": s.HOOK_TL, "scale": scale,
    "pos0_median_norm": pos0, "text_median_norm": rest, "pos0_ratio": pos0 / rest, "eot_mid_window_norm": eot_mid,
    "sink_windows_checked": len(sample), "sinks_pos1": int(big[:, 1].sum()), "sinks_pos2": int(big[:, 2].sum()),
    "sinks_after_skip": late,
    "median_norm_by_position": per_pos, "raw_mean_sq": raw_sq, "scaled_mean_sq": scaled_sq,
    "train_text_tokens": n_train, "train_gb_float32": gb32, "windows": {k: int(len(v)) for k, v in splits.items()}})

print("\nSelf-checks:")
s.check("every window starts with <|endoftext|>", all(int(splits[k][0, 0]) == s.EOT for k in splits))
s.check(f"position 0 is an outlier worth dropping ({pos0 / rest:.0f}x the median text-token norm; bar > 20x)",
        pos0 / rest > 20)
s.check(f"no sink-sized activation after the first {s.SKIP} positions ({late} found)", late == 0)
s.check(f"after scaling, held-out mean |x|² is within 10% of {model.config.hidden_size} ({scaled_sq:.0f})",
        abs(scaled_sq / model.config.hidden_size - 1) < 0.10)
s.check(f"a batch has {cfg['batch']} tokens and no outlier (largest raw norm {max_raw:.0f} < 10x the median {rest:.0f})",
        first.shape[0] == cfg["batch"] and max_raw < 10 * rest)
