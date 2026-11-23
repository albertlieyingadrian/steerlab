"""
Step 3b: train the same SAE with SAELens, the standard library, on exactly the same token windows.

The hand-written trainer in 03a is about 60 lines you can read. SAELens does the same job with more care built in
(activation store, scaling, dead-latent bookkeeping, evals, checkpoints, the file format) and is what most published
SAEs use. Here it gets the same windows, the same layer, the same d_sae, k, learning rate, batch and token count.

Differences worth knowing (SAELens 6.56.1 defaults we keep):
    - its scale is sqrt(d_in) / mean |x| (ours: sqrt(d_in / mean |x|²)); both make activations about sqrt(d_in) long
    - it starts the decoder rows at length 0.1 and divides by their length inside the TopK (rescale_acts_by_decoder_norm)
      instead of forcing them to length 1 after each step
    - its AuxK coefficient is 1.0 on its own normalisation of the aux loss (ours: 1/32 on ours)

Needs SAELens (not in the lab's .venv):  pip install sae-lens==6.56.1
Writes work/<size>/saelens/ (cfg.json + sae_weights.safetensors, scale already folded in) and
results/<size>/train_saelens.json.

    python experiments/sae-101/03b_train_saelens.py --size small
"""

import argparse
import shutil
import time

import sae101 as s

ap = argparse.ArgumentParser()
ap.add_argument("--size", choices=list(s.SIZES), default="small")
ap.add_argument("--max-steps", type=int, default=None)
args = ap.parse_args()
cfg = s.SIZES[args.size]
work = s.WORK / args.size

from datasets import Dataset
from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    LoggingConfig,
    TopKTrainingSAEConfig,
    __version__,
)

train = s.load_tokens(args.size, "train")
steps = s.n_text_tokens(train) // cfg["batch"] - 1
if args.max_steps:
    steps = min(steps, args.max_steps)
dataset = Dataset.from_dict({"input_ids": train.astype("int64").tolist()})  # the same windows 03a reads
device = s.pick_device()
out_dir = work / ("saelens_smoke" if args.max_steps else "saelens")

runner_cfg = LanguageModelSAERunnerConfig(
    sae=TopKTrainingSAEConfig(
        d_in=s.D_MODEL,
        d_sae=cfg["d_sae"],
        k=cfg["k"],
        dtype="float32",
        normalize_activations="expected_average_only_in",
        apply_b_dec_to_input=True,
    ),
    model_name=s.MODEL,
    model_class_name="AutoModelForCausalLM",  # a plain Hugging Face model: no TransformerLens weight processing
    hook_name=s.HOOK_HF,  # the output of block 11: the same activations as 03a
    model_from_pretrained_kwargs={"dtype": "float32"},
    is_dataset_tokenized=True,
    streaming=False,
    context_size=s.CTX,
    prepend_bos=False,  # each window already starts with <|endoftext|>
    disable_concat_sequences=True,  # use each 512-token window as it is
    sequence_separator_token=None,
    seqpos_slice=(s.SKIP, None),  # skip the attention-sink positions, as 03a does
    exclude_special_tokens=[s.EOT],
    training_tokens=steps * cfg["batch"],
    train_batch_size_tokens=cfg["batch"],
    store_batch_size_prompts=32,
    n_sequences_in_buffer=1024,  # about 0.5M tokens, as in 03a
    lr=cfg["lr"],
    lr_scheduler_name="constant",
    lr_warm_up_steps=0,
    lr_decay_steps=int(steps * 0.2),  # linear decay over the last 20%, as in 03a
    adam_beta1=0.9,
    adam_beta2=0.999,
    dead_feature_window=1000,
    device=device,
    seed=s.SEED,
    dtype="float32",
    n_batches_for_norm_estimate=10 if args.max_steps else 100,
    logger=LoggingConfig(log_to_wandb=False),
    output_path=str(out_dir),
    checkpoint_path=str(work / "saelens_checkpoints"),
    n_checkpoints=0,
    verbose=False,
)
print(
    f"1. SAELens {__version__}: TopK, d_sae {cfg['d_sae']:,}, k {cfg['k']}, {steps:,} steps x {cfg['batch']:,} tokens "
    f"on {device}, hook {s.HOOK_HF}"
)
if out_dir.exists():
    shutil.rmtree(out_dir)
t0 = time.monotonic()
sae = LanguageModelSAETrainingRunner(runner_cfg, override_dataset=dataset).run()
minutes = (time.monotonic() - t0) / 60
print(f"2. Done in {minutes:.1f} minutes; saved to work/{args.size}/{out_dir.name}/")
files = sorted(p.name for p in out_dir.rglob("*") if p.is_file())
print(f"   files: {files}")
s.save_json(
    s.RESULTS / args.size / ("train_saelens_smoke.json" if args.max_steps else "train_saelens.json"),
    {
        "sae_lens_version": __version__,
        "d_sae": cfg["d_sae"],
        "k": cfg["k"],
        "lr": cfg["lr"],
        "steps": steps,
        "tokens": steps * cfg["batch"],
        "minutes": minutes,
        "device": device,
        "files": files,
    },
)

print("\nSelf-checks:")
s.check("SAELens wrote cfg.json and sae_weights.safetensors", {"cfg.json", "sae_weights.safetensors"} <= set(files))
saved = s.read_json(out_dir / "cfg.json")
s.check(
    f"the saved SAE takes raw activations (normalize_activations = {saved.get('normalize_activations')!r})",
    saved.get("normalize_activations") == "none",
)
s.check(
    f"architecture {saved.get('architecture')!r}, k {saved.get('k')}, d_sae {saved.get('d_sae')}",
    saved.get("architecture") == "topk" and saved.get("k") == cfg["k"] and saved.get("d_sae") == cfg["d_sae"],
)
