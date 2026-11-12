"""
The follow-up study's GPU work on Modal: fine-tune the `sft` arm, then sample every arm on the evaluation subjects.

COSTS MONEY (a few dollars in all; every function has a timeout). Run from the repo root:
    # 1. fine-tune on Claude Opus 5.5's sketches (results/next/sft_train.jsonl, from script 12)
    .venv/bin/modal run experiments/pigment-gym/modal_next.py --stage sft
    # 2. sample 8 groups of 8 per evaluation subject, render, score with Pigment Gym's reward
    .venv/bin/modal run experiments/pigment-gym/modal_next.py --stage sample --arm base
    .venv/bin/modal run experiments/pigment-gym/modal_next.py --stage sample --arm grpo   # the laptop run's LoRA
    .venv/bin/modal run experiments/pigment-gym/modal_next.py --stage sample --arm sft
    # add --smoke for 1 subject x 1 group (or 16 examples x 1 epoch for --stage sft)

Every model runs in float32 (as the laptop run did), with the same seeds per (subject, group), so arms differ only in
their weights. Sample rows go to results/next/samples_<arm>.jsonl; the renders that can be judged (sample 0 of each
group, and the group's highest-reward sample) go to the work folder next/png/<hash>.png. Generation runs on Modal;
rendering and scoring run on the laptop, because Chromium's software WebGL on Modal's CPUs times out (debug_render).

Cost at list price: an L40S is about $1.95/h plus CPU and memory; an H100 about $4.59/h.
"""

import json
import os
import sys
from pathlib import Path

import modal

PART_DIR = Path(__file__).resolve().parent
REMOTE_PART = "/root/pigment-gym"
VOLUME_PATH = "/vol"

image = (
    modal.Image.from_registry("python:3.11-slim-bookworm")
    .pip_install(
        "torch==2.14.1",
        "transformers==5.20.2",
        "peft==0.21.2",
        "accelerate==1.15.0",
        "open_clip_torch==3.3.0",
        "playwright==1.64.1",
        "python-dotenv",
        "numpy",
        "pillow",
    )
    .run_commands("playwright install --with-deps chromium")
    .env({"HF_HOME": f"{VOLUME_PATH}/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(PART_DIR, REMOTE_PART, ignore=["**/__pycache__", "results", "report", "sketches"])
)
volume = modal.Volume.from_name("pigment-gym-next", create_if_missing=True)
app = modal.App("pigment-gym-next", image=image)


def _write_adapter(files, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (folder / name).write_bytes(data)
    return folder


def _load_model(adapter_files, merge_files=None):
    import tempfile

    import torch
    from next_common import MODEL
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to("cuda")
    if merge_files:  # sft_grpo: the fine-tuned LoRA merged in first, then the GRPO LoRA on top
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, str(_write_adapter(merge_files, Path(tempfile.mkdtemp())))
        ).merge_and_unload()
    if adapter_files:
        from peft import PeftModel

        folder = _write_adapter(adapter_files, Path(tempfile.mkdtemp()))
        model = PeftModel.from_pretrained(model, str(folder)).to("cuda")
    return model.eval(), tokenizer


@app.function(gpu="H100", cpu=4.0, memory=32768, timeout=40 * 60, volumes={VOLUME_PATH: volume})
def sft(rows, epochs=3, learning_rate=2e-4, batch=8, micro_batch=2, seed=0):
    """LoRA fine-tuning on (prompt, reply) pairs, loss on the reply tokens only. Returns the adapter and a loss log."""
    import random
    import tempfile
    import time

    import torch

    sys.path[:0] = [REMOTE_PART]
    from brush_tools import SYSTEM_PROMPT
    from grpo_tools import build_lora_model, chat_prompt
    from next_common import MODEL

    started = time.perf_counter()
    model, tokenizer = build_lora_model(MODEL, "cuda", seed=seed)
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    examples = []
    for row in rows:
        prompt = chat_prompt(
            tokenizer,
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Paint {row['subject']}."}],
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        reply_ids = tokenizer(row["reply"], add_special_tokens=False)["input_ids"] + [end_id]
        examples.append((prompt_ids, reply_ids))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    rng = random.Random(seed)
    log = []
    model.train()
    for epoch in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for start in range(0, len(order), batch):
            chunk = [examples[i] for i in order[start : start + batch]]
            total_reply = sum(len(r) for _, r in chunk)
            optimizer.zero_grad()
            loss_sum = 0.0
            for m in range(0, len(chunk), micro_batch):
                part = chunk[m : m + micro_batch]
                length = max(len(p) + len(r) for p, r in part)
                # right padding for training: a padded position is masked out of attention and of the loss
                ids = torch.full((len(part), length), tokenizer.pad_token_id, dtype=torch.long)
                labels = torch.full((len(part), length), -100, dtype=torch.long)
                attention = torch.zeros((len(part), length), dtype=torch.long)
                for row_i, (p, r) in enumerate(part):
                    sequence = p + r
                    ids[row_i, : len(sequence)] = torch.tensor(sequence)
                    labels[row_i, len(p) : len(sequence)] = torch.tensor(r)
                    attention[row_i, : len(sequence)] = 1
                ids, labels, attention = ids.cuda(), labels.cuda(), attention.cuda()
                logits = model(input_ids=ids, attention_mask=attention).logits[:, :-1].float()
                loss = (
                    torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        labels[:, 1:].reshape(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    / total_reply
                )
                loss.backward()
                loss_sum += float(loss.detach())
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            log.append({"epoch": epoch, "step": len(log), "loss": loss_sum})
            print(f"epoch {epoch} step {len(log)} loss {loss_sum:.4f}", flush=True)
    folder = Path(tempfile.mkdtemp())
    model.save_pretrained(folder)
    files = {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file() and p.suffix in (".json", ".safetensors")}
    return {"files": files, "log": log, "seconds": time.perf_counter() - started, "examples": len(examples)}


@app.function(gpu="L40S", cpu=4.0, memory=32768, timeout=45 * 60, volumes={VOLUME_PATH: volume})
def sample_subject(arm, adapter_files, subject_index, groups, merge_files=None):
    """8 x `groups` sketches for one evaluation subject. Generation only: rendering happens on the laptop, because
    Chromium's software WebGL on Modal's CPUs times out on ordinary sketches (see debug_render)."""
    import time

    sys.path[:0] = [REMOTE_PART]
    from brush_tools import SYSTEM_PROMPT, extract_code
    from grpo_tools import chat_prompt, sample_group
    from next_common import EVAL_SUBJECTS, GROUP_SIZE, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, group_seed

    started = time.perf_counter()
    subject = EVAL_SUBJECTS[subject_index]
    model, tokenizer = _load_model(adapter_files, merge_files)
    prompt = chat_prompt(
        tokenizer, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Paint {subject}."}]
    )
    rows = []
    for group in range(groups):
        texts, _, mask = sample_group(
            model,
            tokenizer,
            prompt,
            GROUP_SIZE,
            MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            seed=group_seed(subject_index, group),
        )
        for slot, text in enumerate(texts):
            rows.append(
                {
                    "arm_model": arm,
                    "subject": subject,
                    "subject_index": subject_index,
                    "group": group,
                    "slot": slot,
                    "tokens": int(mask[slot].sum()),
                    "code": extract_code(text),
                }
            )
        print(f"{arm} {subject!r} group {group} done", flush=True)
    volume.commit()  # keeps the downloaded model cached for the next container
    return {"rows": rows, "seconds": time.perf_counter() - started}


def render_and_score(rows, png_dir):
    """On the laptop: render every sketch, score it with Pigment Gym's reward (caption = its subject, copy check on),
    mark each group's highest-reward sketch, and save the PNGs of sample 0 and the best of each group."""
    from brush_tools import PROMPT_EXAMPLE, Renderer, extract_code
    from next_common import png_key
    from reward_tools import Scorer, score_rollouts, similarity

    example = extract_code(PROMPT_EXAMPLE)
    scorer = Scorer()
    groups = {}
    for row in rows:
        groups.setdefault((row["subject_index"], row["group"]), []).append(row)
    out = []
    with Renderer(pool_size=8) as renderer:
        for key in sorted(groups):
            group = sorted(groups[key], key=lambda r: r["slot"])
            codes = [r["code"] for r in group]
            renders = renderer.render_many(codes)
            scored = score_rollouts(codes, renders, scorer, group[0]["subject"], forbid_copy_of=example)
            best = max(range(len(scored)), key=lambda i: scored[i]["reward"])
            for slot, (row, score, render) in enumerate(zip(group, scored, renders, strict=True)):
                keep = slot in (0, best) and render.png is not None
                name = png_key(render.png) if keep else None
                if keep:
                    (png_dir / f"{name}.png").write_bytes(render.png)
                code = row.pop("code")
                out.append(
                    {
                        **row,
                        "is_best": slot == best,
                        "similarity": similarity(example, code),
                        "png": name,
                        "render_s": render.seconds,
                        **score,
                        "code": code,
                    }
                )
    return out


@app.function(cpu=4.0, memory=8192, timeout=10 * 60)
def debug_render(codes, pool_size=1):
    """Render on Modal's CPU image and return each render's errors, coverage and time (renderer diagnosis)."""
    sys.path[:0] = [REMOTE_PART]
    from brush_tools import Renderer, gate

    with Renderer(pool_size=pool_size) as renderer:
        renders = renderer.render_many(codes)
    return [
        {
            "errors": r.errors[:3],
            "finished": r.finished,
            "coverage": r.coverage,
            "seconds": r.seconds,
            "violations": gate(c, r).violations,
        }
        for c, r in zip(codes, renders, strict=True)
    ]


@app.local_entrypoint()
def main(stage: str = "sample", arm: str = "base", smoke: bool = False):
    sys.path[:0] = [str(PART_DIR)]
    from common import RESULTS, outputs_dir
    from next_common import EVAL_SUBJECTS, GROUPS

    out = RESULTS / "next"
    work = outputs_dir() / "next"
    (work / "png").mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if smoke else ""
    if stage == "sft":
        rows = [json.loads(line) for line in (out / "sft_train.jsonl").read_text().splitlines()]
        result = sft.remote(rows[:16] if smoke else rows, epochs=1 if smoke else 3)
        _write_adapter(result["files"], work / f"sft_lora{tag}")
        (out / f"sft_log{tag}.json").write_text(
            json.dumps({"examples": result["examples"], "seconds": result["seconds"], "log": result["log"]}, indent=1)
        )
        print(
            f"fine-tuned on {result['examples']} sketches in {result['seconds'] / 60:.1f} min; "
            f"loss {result['log'][0]['loss']:.3f} -> {result['log'][-1]['loss']:.3f}; adapter in {work}/sft_lora{tag}"
        )
        return
    # The GRPO LoRA is the laptop run's (7_train_laptop.py saves it to the work folder's lora/; not in git).
    # PIGMENT_GRPO_LORA points elsewhere, e.g. the copy this note used.
    grpo = Path(os.environ.get("PIGMENT_GRPO_LORA", outputs_dir() / "lora"))
    folder = {"base": None, "grpo": grpo, "sft": work / f"sft_lora{tag}", "sft_grpo": work / "sft_grpo" / "lora"}[arm]
    files = None
    if folder is not None:
        files = {p.name: p.read_bytes() for p in folder.iterdir() if p.suffix in (".json", ".safetensors")}
    merge = None
    if arm == "sft_grpo":
        merge = {p.name: p.read_bytes() for p in (work / "sft_lora").iterdir() if p.suffix in (".json", ".safetensors")}
    subjects = [0] if smoke else list(range(len(EVAL_SUBJECTS)))
    groups = 1 if smoke else GROUPS
    rows, seconds = [], 0.0
    n = len(subjects)
    for result in sample_subject.map([arm] * n, [files] * n, subjects, [groups] * n, [merge] * n):
        rows += result["rows"]
        seconds += result["seconds"]
    print(
        f"generated {len(rows)} sketches in {seconds / 60:.1f} L40S container-minutes "
        f"(about ${seconds / 3600 * 2.3:.2f} at list price incl. CPU and memory); rendering on this machine"
    )
    rows = render_and_score(rows, work / "png")
    rows.sort(key=lambda r: (r["subject_index"], r["group"], r["slot"]))
    with (out / f"samples_{arm}{tag}.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"{arm}: {len(rows)} sketches, {sum(r['gate'] for r in rows):.0f} pass the gate")
