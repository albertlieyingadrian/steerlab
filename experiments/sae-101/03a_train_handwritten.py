"""
Step 3a: train the hand-written TopK SAE (sae101.TopKSAE) on the residual stream, streaming activations through the
shuffle buffer from step 2.

What happens per step: take 4,096 shuffled activations, encode, keep the k biggest latents per token, decode, and
nudge every weight to shrink the squared error (Adam). After each step the decoder rows are set back to length 1.
AuxK gives latents that have not fired for 1,000 steps a small job (rebuild what the live latents missed), so they
can come back to life. The decoder bias starts at the mean activation.

Writes work/<size>/sae_handwritten.pt (and a checkpoint it can resume from) and results/<size>/train_handwritten.json.

    .venv/bin/python experiments/sae-101/03a_train_handwritten.py --size small     # about 10 minutes on a laptop
    (the full size runs on one A10G: modal_app.py)
"""

import argparse
import time

import torch

import sae101 as s

ap = argparse.ArgumentParser()
ap.add_argument("--size", choices=list(s.SIZES), default="small")
ap.add_argument("--max-steps", type=int, default=None, help="stop early (smoke tests)")
args = ap.parse_args()
cfg = s.SIZES[args.size]
work = s.WORK / args.size
torch.manual_seed(s.SEED)

model, _, device = s.load_lm()
train = s.load_tokens(args.size, "train")
stats = torch.load(work / "stats.pt", weights_only=False)
scale = stats["scale"]
total_steps = s.n_text_tokens(train) // cfg["batch"] - 1
if args.max_steps:
    total_steps = min(total_steps, args.max_steps)

sae = s.TopKSAE(model.config.hidden_size, cfg["d_sae"], cfg["k"]).to(device)
with torch.no_grad():
    sae.b_dec.copy_(stats["mean_act"].to(device) * scale)  # start the "average activation" at the data's average

# The checkpoint lets an interrupted run pick up where it stopped. It records the settings it was trained with, and it
# is resumed only when they match this run, so a rerun with a changed k or lr starts over instead of quietly
# continuing the old weights. It is deleted once the final SAE is saved, so a rerun after a finished run retrains.
ckpt_path = work / "ckpt_handwritten.pt"
run_cfg = {"d_sae": cfg["d_sae"], "k": cfg["k"], "lr": cfg["lr"], "batch": cfg["batch"], "total_steps": total_steps}
start_step, start_window, opt_state, last_fired, log, minutes_before = 0, 0, None, None, [], 0.0
if ckpt_path.exists() and not args.max_steps:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ck.get("run_cfg") == run_cfg:
        sae.load_state_dict(ck["sae"])
        start_step, start_window, opt_state, last_fired, log = (
            ck["step"],
            ck["windows"],
            ck["opt"],
            ck["last_fired"],
            ck["log"],
        )
        minutes_before = ck.get("minutes", 0.0)
        print(f"   resuming from step {start_step:,} (window {start_window:,})")
    else:
        print(f"   ignoring {ckpt_path.name}: it was saved with other settings ({ck.get('run_cfg')}), starting over")

print(
    f"1. Training a TopK SAE: d_in {sae.d_in} -> d_sae {sae.d_sae:,} latents ({sae.d_sae / sae.d_in:.1f}x), "
    f"k = {sae.k} per token, on {device}"
)
print(
    f"   {total_steps:,} steps x {cfg['batch']:,} tokens = {total_steps * cfg['batch'] / 1e6:,.1f}M tokens; Adam lr "
    f"{cfg['lr']:g}, falling to 0 over the last 20%"
)


def on_log(row):
    log.append(row)
    if row["step"] % max(100, total_steps // 10 // 100 * 100) == 0 or row["step"] == total_steps:
        print(
            f"   step {row['step']:>6,}  tokens {row['tokens'] / 1e6:>6.1f}M  train FVU {row['train_fvu']:.4f}  "
            f"dead {row['dead_share']:.1%}  lr {row['lr']:.1e}  {row['seconds']:>6.0f}s",
            flush=True,
        )


def on_checkpoint(step, opt, fired, windows):
    if not args.max_steps:
        torch.save(
            {
                "sae": sae.state_dict(),
                "step": step,
                "windows": windows,
                "opt": opt.state_dict(),
                "last_fired": fired,
                "log": log,
                "run_cfg": run_cfg,
                "minutes": minutes_before + (time.monotonic() - t0) / 60,
            },
            ckpt_path,
        )


t0 = time.monotonic()
batches = s.activation_batches(model, train, batch=cfg["batch"], scale=scale, device=device, start=start_window)
s.train_sae(
    sae,
    batches,
    total_steps=total_steps,
    lr=cfg["lr"],
    on_log=on_log,
    on_checkpoint=on_checkpoint,
    start_step=start_step,
    opt_state=opt_state,
    last_fired=last_fired,
)
minutes = minutes_before + (time.monotonic() - t0) / 60  # includes earlier segments when the run was resumed
name = "sae_handwritten_smoke.pt" if args.max_steps else "sae_handwritten.pt"
s.save_sae(sae, work / name, {"scale": scale, "trainer": "hand-written", "size": args.size})
if not args.max_steps:
    ckpt_path.unlink(missing_ok=True)  # the run is finished; a rerun should train afresh, not resume at the last step
s.save_json(
    s.RESULTS / args.size / ("train_handwritten_smoke.json" if args.max_steps else "train_handwritten.json"),
    {
        "d_sae": sae.d_sae,
        "k": sae.k,
        "lr": cfg["lr"],
        "batch": cfg["batch"],
        "steps": total_steps,
        "tokens": total_steps * cfg["batch"],
        "minutes": minutes,
        "device": device,
        "log": log,
    },
)
print(f"\n2. Done in {minutes:.1f} minutes; wrote work/{args.size}/{name}")

norms = sae.W_dec.detach().norm(dim=1)
last = log[-1]
early = log[0]
print("\nSelf-checks:")
s.check(
    f"decoder rows have length 1 (min {float(norms.min()):.4f}, max {float(norms.max()):.4f})",
    float((norms - 1).abs().max()) < 1e-4,
)
s.check(
    f"train FVU fell from {early['train_fvu']:.3f} (step {early['step']}) to {last['train_fvu']:.3f} "
    f"(step {last['step']})",
    last["train_fvu"] < early["train_fvu"],
)
s.check("no NaN or inf in the weights", all(bool(torch.isfinite(p).all()) for p in sae.parameters()))
s.check(f"exactly k = {sae.k} latents active per token (train L0 {last['l0']:.2f})", abs(last["l0"] - sae.k) < 0.5)
