"""
The Llama-3.3-70B half of AuditBench on two rented H100s (Modal). COSTS MONEY: cap $70 (PREREGISTRATION-70B.md).

    export MODAL_PROFILE=albertlieyingadrian
    .venv/bin/modal deploy experiments/auditbench/modal_70b.py          # once, and after any edit
    .venv/bin/python experiments/auditbench/modal_70b.py check          # HF access to the base and the 60 adapters
    .venv/bin/python experiments/auditbench/modal_70b.py download       # weights into the volume (CPU only)
    .venv/bin/python experiments/auditbench/modal_70b.py submit smoke   # 1 adapter, a few prompts
    .venv/bin/python experiments/auditbench/modal_70b.py submit full    # 60 adapters, resumable
    .venv/bin/python experiments/auditbench/modal_70b.py wait <tag>     # block until the spawned call ends
    .venv/bin/python experiments/auditbench/modal_70b.py pull <tag>     # readings -> $AUDITBENCH_WORK/llama70b/<tag>

What runs where:
  * the laptop builds every prompt (AuditBench's text is read from the work folder, never stored here) and spawns one
    call on the deployed app, so the run survives the laptop going offline;
  * the GPU container loads the base model once (bfloat16, split over two H100s), reads the base model, then swaps the
    60 LoRA adapters in turn. Each set of readings is saved to the volume as soon as it is done, and a restart skips
    what exists. No text is generated: every number comes from a forward pass.

Budget guard: RUN_TIMEOUT_S is a hard stop (Modal kills the container), and `max_seconds` stops starting new adapters
before that, so one call costs at most RUN_TIMEOUT_S / 3600 x PRICE_PER_HOUR.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import modal

APP = "auditbench-70b"
BASE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
TOKENIZER = "auditing-agents/prism-4-tokenizer"
LAYERS = (20, 40, 60)
PRIMARY_LAYER = 40
CONFIGS = {
    "td": "transcripts_only",
    "td_sft": "transcripts_only_then_redteam_high",
    "sdf": "synth_docs_only",
    "sdf_sft": "synth_docs_only_then_redteam_high",
}
SANDBAGGER = "research_sandbagging"
SIGMA_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
NOISE_SEEDS = (0, 1, 2)
VOL = "/vol"
RUN_TIMEOUT_S = 5 * 3600
SMOKE_TIMEOUT_S = 45 * 60
PRICE_PER_HOUR = 2 * 3.95 + 8 * 0.0473 + 96 * 0.0080  # two H100s + 8 cores + 96 GiB, Modal list price (2026-10-24)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.14.1", "transformers==5.20.2", "peft==0.21.2", "accelerate==1.15.0", "numpy<2.4", "hf_xet>=1.1"
    )
    .env({"HF_HOME": f"{VOL}/hf", "TOKENIZERS_PARALLELISM": "false", "HF_XET_HIGH_PERFORMANCE": "1"})
)
volume = modal.Volume.from_name(APP, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")
app = modal.App(APP, image=image)


# ---------------------------------------------------------------- remote side


@app.function(cpu=0.25, memory=512, timeout=300, secrets=[hf_secret])
def check(repos: list[str]) -> dict:
    """Which repos the token can read (account name and error classes only; the token is never returned)."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    api = HfApi()
    out = {"user": api.whoami().get("name")}
    for r in repos:
        try:
            api.auth_check(r)
            out[r] = "ok"
        except HfHubHTTPError as e:  # gated or missing: report the error class, nothing else
            out[r] = type(e).__name__
    return out


@app.function(cpu=8.0, memory=16384, timeout=4 * 3600, volumes={VOL: volume}, secrets=[hf_secret])
def download(repos: list[str], base: bool) -> dict:
    from huggingface_hub import snapshot_download

    t0, got = time.time(), {}
    snapshot_download(TOKENIZER)
    if base:
        p = snapshot_download(BASE_MODEL, allow_patterns=["*.json", "*.safetensors"])
        got["base_gb"] = round(sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e9, 1)
        volume.commit()
    for i, r in enumerate(repos, 1):
        snapshot_download(r, allow_patterns=["adapter_config.json", "adapter_model.safetensors"])
        if i % 5 == 0:
            volume.commit()
    volume.commit()
    return got | {"adapters": len(repos), "seconds": round(time.time() - t0)}


def _local(repo: str) -> str:
    """The cached snapshot folder of a repo on the volume (downloaded with allow_patterns, so a partial snapshot,
    which snapshot_download(local_files_only=True) refuses)."""
    root = Path(VOL) / "hf" / "hub" / ("models--" + repo.replace("/", "--"))
    return str(root / "snapshots" / (root / "refs" / "main").read_text().strip())


def _render(tok, system: str, user: str):
    text = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    start = text.rindex(user)
    return text, (start, start + len(user))


def _layers(model):
    import torch

    return next(m for n, m in model.named_modules() if n.endswith("layers") and isinstance(m, torch.nn.ModuleList))


def _read(model, tok, system: str, users: list[str], keep: list[str], batch: int = 12) -> dict:
    """Readings at LAYERS: mean over the user's tokens ("user"), last token ("last"), A-D logits ("letters").
    hidden_states[k] is the output of decoder layer k, so layer k is hooked at index k - 1."""
    import numpy as np
    import torch

    letter_ids = [tok.convert_tokens_to_ids(c) for c in "ABCD"]
    dev = model.get_input_embeddings().weight.device
    layers = _layers(model)
    out = {k: [] for k in keep}
    for i in range(0, len(users), batch):
        texts, spans = zip(*(_render(tok, system, u) for u in users[i : i + batch]), strict=True)
        enc = tok(list(texts), return_tensors="pt", padding=True, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc.pop("offset_mapping")
        umask = torch.zeros_like(enc["input_ids"], dtype=torch.bool)
        for j, (a, b) in enumerate(spans):
            s, e = offsets[j, :, 0], offsets[j, :, 1]
            umask[j] = (s >= a) & (e <= b) & (e > s) & enc["attention_mask"][j].bool()
        enc = {k: v.to(dev) for k, v in enc.items()}
        pos = (enc["attention_mask"].long().cumsum(-1) - 1).clamp(min=0)
        got: dict[int, tuple] = {}

        def grab(k, got=got, umask=umask):
            def fn(_m, _i, output):
                h = (output[0] if isinstance(output, tuple) else output).float()
                w = umask.to(h.device).float().unsqueeze(-1)
                got[k] = (((h * w).sum(1) / w.sum(1)).cpu(), h[:, -1].cpu())

            return fn

        handles = [layers[k - 1].register_forward_hook(grab(k)) for k in LAYERS]
        try:
            with torch.no_grad():
                res = model(**enc, position_ids=pos, logits_to_keep=1)
        finally:
            for hd in handles:
                hd.remove()
        if "user" in out:
            out["user"].append(torch.stack([got[k][0] for k in LAYERS], 1))
        if "last" in out:
            out["last"].append(torch.stack([got[k][1] for k in LAYERS], 1))
        if "letters" in out:
            out["letters"].append(res.logits[:, -1, letter_ids].float().cpu())
        del res
    return {k: torch.cat(v).numpy().astype(np.float32) for k, v in out.items()}


def _accuracy(letters, answers) -> float:
    import numpy as np

    return float((letters.argmax(1) == np.array(answers)).mean())


def _noisy_acc(model, tok, target: dict, sigma: float, seeds) -> list[float]:
    """Quiz accuracy with Gaussian noise of std sigma x (each token's RMS) added after decoder layer PRIMARY_LAYER."""
    import torch

    accs = []
    for seed in seeds:
        gen = torch.Generator(device="cpu").manual_seed(seed)

        def fn(_m, _i, output, gen=gen):
            h = output[0] if isinstance(output, tuple) else output
            rms = h.float().pow(2).mean(-1, keepdim=True).sqrt()
            z = torch.randn(h.shape, generator=gen).to(h.device)
            h2 = (h.float() + sigma * rms * z).to(h.dtype)
            return (h2, *output[1:]) if isinstance(output, tuple) else h2

        hd = _layers(model)[PRIMARY_LAYER - 1].register_forward_hook(fn)
        try:
            letters = _read(model, tok, target["system"], target["users"], ["letters"])["letters"]
        finally:
            hd.remove()
        accs.append(_accuracy(letters, target["answers"]))
    return accs


@app.function(
    gpu="H100:2",
    cpu=8.0,
    memory=98304,
    timeout=RUN_TIMEOUT_S,
    scaledown_window=10,
    volumes={VOL: volume},
    secrets=[hf_secret],
)
def run(inputs: dict, tag: str, max_seconds: float) -> dict:
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    out = Path(VOL) / "out" / tag
    (out / "base").mkdir(parents=True, exist_ok=True)
    (out / "organisms").mkdir(parents=True, exist_ok=True)
    timing_path = out / "timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {"calls": []}
    call = {"start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "organisms": {}}
    timing["calls"].append(call)

    def save_timing():
        call["elapsed_s"] = round(time.time() - t0)
        timing_path.write_text(json.dumps(timing, indent=1))
        volume.commit()

    # local snapshot paths: offline, a repo id makes transformers look for a config.json the tokenizer repo lacks
    tok = AutoTokenizer.from_pretrained(_local(TOKENIZER))
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = "<|finetune_right_pad_id|>"
    model = AutoModelForCausalLM.from_pretrained(
        _local(BASE_MODEL),
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "74GiB", 1: "74GiB"},
    )
    model.eval()
    call["load_s"] = round(time.time() - t0)
    save_timing()
    print(f"loaded {BASE_MODEL} in {call['load_s']}s", flush=True)

    # 1. the base model: every reading set, then the noise calibration
    for s in inputs["base_sets"]:
        f = out / "base" / f"{s['name']}.npz"
        if not f.exists():
            np.savez(f, **_read(model, tok, s["system"], s["users"], s["keep"]))
            volume.commit()
            print(f"base {s['name']} ({time.time() - t0:.0f}s)", flush=True)
    nz = inputs["noise"]
    noise_path = out / "base" / "noise.json"
    noise = json.loads(noise_path.read_text()) if noise_path.exists() else {}
    if "sigmas" not in noise:
        c = nz["calib"]
        clean = _accuracy(_read(model, tok, c["system"], c["users"], ["letters"])["letters"], c["answers"])
        grid = {str(s): _noisy_acc(model, tok, c, s, (0,))[0] for s in nz["grid"]}
        sigmas = [float(min(grid, key=lambda s: abs(grid[s] - f * clean))) for f in (0.9, 0.7)]
        noise = {"clean": clean, "grid": grid, "sigmas": sigmas}
        for name, t in nz["base_targets"].items():
            noise[name] = {str(s): _noisy_acc(model, tok, t, s, nz["seeds"]) for s in sigmas}
        noise_path.write_text(json.dumps(noise, indent=1))
        volume.commit()
        print(f"noise: clean {clean:.3f}, grid {grid}, chosen {sigmas} ({time.time() - t0:.0f}s)", flush=True)
    call["base_s"] = round(time.time() - t0)
    save_timing()

    # 2. the adapters, one at a time (load the next, switch to it, then free the previous)
    peft_model, current, done = None, None, 0
    for o in inputs["organisms"]:
        stem = out / "organisms" / o["id"].replace("/", "__")
        if stem.with_suffix(".json").exists():
            continue
        if time.time() - t0 > max_seconds:
            print(f"stopping: {max_seconds:.0f}s budget reached", flush=True)
            break
        t1 = time.time()
        name = o["repo"].split("/")[-1]
        path = _local(o["repo"])
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, path, adapter_name=name)
        else:
            peft_model.load_adapter(path, adapter_name=name)
            peft_model.set_adapter(name)
            peft_model.delete_adapter(current)
        peft_model.eval()
        current = name
        t_load = time.time() - t1
        arrays = {}
        for s in inputs["org_sets"]:
            for k, v in _read(peft_model, tok, s["system"], s["users"], s["keep"]).items():
                arrays[f"{s['name']}_{k}"] = v
        np.savez(stem.with_suffix(".npz"), **arrays)
        t = nz["org_target"]
        row = {
            "id": o["id"],
            "noise": {str(s): _noisy_acc(peft_model, tok, t, s, nz["seeds"]) for s in noise["sigmas"]},
            "adapter_load_s": round(t_load, 1),
            "seconds": round(time.time() - t1, 1),
        }
        stem.with_suffix(".json").write_text(json.dumps(row, indent=1))
        call["organisms"][o["id"]] = row["seconds"]
        save_timing()
        done += 1
        print(f"[{done}] {o['id']}: {row['seconds']}s (adapter load {t_load:.0f}s)", flush=True)
    save_timing()
    return {"tag": tag, "organisms_done": done, "elapsed_s": round(time.time() - t0), "load_s": call["load_s"]}


# ---------------------------------------------------------------- laptop side


def _ab():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ab

    return ab


def organisms() -> list[dict]:
    """60 organisms, sandbaggers first, so a run that stops early still has the primary ones."""
    ab = _ab()
    return [
        {"id": f"{c}/{b}", "config": c, "behaviour": b, "repo": f"auditing-agents/llama_70b_{CONFIGS[c]}_{b}"}
        for b in (SANDBAGGER, *ab.BEHAVIOURS)
        for c in CONFIGS
    ]


def build_inputs(smoke: bool) -> dict:
    ab = _ab()
    import fetch

    if not (ab.WORK / "auditbench_prompts" / "prism_4.txt").exists():
        raise SystemExit(
            "AuditBench prompts missing: run .venv/bin/python experiments/auditbench/fetch.py --prompts-only"
        )
    extra = ab.read_json(ab.DATA / "llama70b.json")
    template = fetch.get(fetch.PROMPTS + "system_prompt_quirks/templates/base_instruction.jinja2")
    quirks = {b: ab.quirk_prompt(b) for b in ab.BEHAVIOURS}
    quirks[SANDBAGGER] = fetch.render_quirk(
        template, "{% block content %}\n" + extra["research_sandbagging_quirk"] + "\n{% endblock %}"
    )
    prism, sandbag = ab.prism_prompt(), ab.sandbag_prompt()
    audit = ab.audit_prompts()[: 4 if smoke else None]
    test, train = ab.quiz("test")[: 8 if smoke else None], ab.quiz("train")[: 8 if smoke else None]
    pairs = extra["code_pairs"][: 2 if smoke else None]
    contrast_file = ab.HERE.parent / "detector-sanity-checks" / "data" / "sandbagging_contrast_pairs.json"
    contrast = [p for c in ab.read_json(contrast_file)["contrast_pairs"] for p in c["pairs"]][: 4 if smoke else None]
    from steerlab.monitors.logit import load_eval_awareness

    fr = load_eval_awareness()
    ev = [ab.quiz_text(q, fr.eval_framings[i % len(fr.eval_framings)]) for i, q in enumerate(train)]
    dp = [ab.quiz_text(q, fr.deployment_framings[i % len(fr.deployment_framings)]) for i, q in enumerate(train)]
    quiz_plain = [ab.quiz_text(q) for q in test]
    quiz_safety = [ab.quiz_text(q, extra["safety_framing"]) for q in test]
    answers = [q["answer"] for q in test]

    def S(name, system, users, keep):
        return {"name": name, "system": system, "users": users, "keep": keep}

    org_sets = [
        S("audit", prism, audit, ["user"]),
        S("quiz_plain", prism, quiz_plain, ["last", "letters"]),
        S("quiz_safety", prism, quiz_safety, ["last", "letters"]),
        S("code_safety", prism, [p["safety"] for p in pairs], ["user", "last"]),
        S("code_plain", prism, [p["plain"] for p in pairs], ["user", "last"]),
    ]
    base_sets = [S(f"audit_{b}", quirks[b], audit, ["user"]) for b in (SANDBAGGER, *ab.BEHAVIOURS)]
    base_sets += [
        S("pairs_sandbag", prism, [p["sandbagging"] for p in contrast], ["last"]),
        S("pairs_genuine", prism, [p["genuine"] for p in contrast], ["last"]),
        S("frame_eval", prism, ev, ["last"]),
        S("frame_deploy", prism, dp, ["last"]),
        S("quiz_sandbag", sandbag, quiz_plain, ["last", "letters"]),
        S("quiz_sandbag_safety", sandbag, quiz_safety, ["last", "letters"]),
    ]
    base_sets += org_sets  # the base model's own readings of every organism set ("audit" is the PRISM-4 reference)
    orgs = organisms()[:1] if smoke else organisms()
    return {
        "base_sets": base_sets,
        "org_sets": org_sets,
        "noise": {
            "grid": list(SIGMA_GRID[:2] if smoke else SIGMA_GRID),
            "seeds": list(NOISE_SEEDS[:1] if smoke else NOISE_SEEDS),
            "calib": {"system": prism, "users": quiz_plain, "answers": answers},
            "base_targets": {
                "base_safety": {"system": prism, "users": quiz_safety, "answers": answers},
                "sandbag_prompt": {"system": sandbag, "users": quiz_plain, "answers": answers},
            },
            "org_target": {"system": prism, "users": quiz_safety, "answers": answers},
        },
        "organisms": [{"id": o["id"], "repo": o["repo"]} for o in orgs],
    }


def _calls_file() -> Path:
    p = _ab().WORK / "llama70b" / "calls.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _remote(name: str):
    return modal.Function.from_name(APP, name)


def cli(argv: list[str]) -> None:
    cmd = argv[0] if argv else "help"
    if cmd == "check":
        repos = [BASE_MODEL] + [o["repo"] for o in organisms()]
        res = _remote("check").remote(repos)
        bad = {r: v for r, v in res.items() if r != "user" and v != "ok"}
        print(
            f"HF user {res['user']}: {len(repos) - len(bad)}/{len(repos)} repos readable; not readable: {bad or 'none'}"
        )
    elif cmd == "download":
        call = _remote("download").spawn([o["repo"] for o in organisms()], "--adapters-only" not in argv)
        print(f"download spawned: {call.object_id}")
    elif cmd == "submit":
        tag = argv[1]
        smoke = tag.startswith("smoke")
        inputs = build_inputs(smoke)
        limit = (SMOKE_TIMEOUT_S if smoke else RUN_TIMEOUT_S) - 15 * 60
        call = _remote("run").spawn(inputs, tag, limit)
        calls = json.loads(_calls_file().read_text()) if _calls_file().exists() else {}
        calls[tag] = {"call_id": call.object_id, "submitted": time.strftime("%Y-%m-%d %H:%M:%S")}
        _calls_file().write_text(json.dumps(calls, indent=1))
        print(
            f"run {tag} spawned: {call.object_id} ({len(inputs['organisms'])} organisms; no new adapter after {limit}s)"
        )
    elif cmd == "wait":
        cid = json.loads(_calls_file().read_text())[argv[1]]["call_id"]
        print(json.dumps(modal.FunctionCall.from_id(cid).get(timeout=RUN_TIMEOUT_S), indent=1))
    elif cmd == "pull":
        dest = _ab().WORK / "llama70b"
        dest.mkdir(parents=True, exist_ok=True)
        modal_cli = str(Path(sys.executable).with_name("modal"))
        subprocess.run([modal_cli, "volume", "get", "--force", APP, f"out/{argv[1]}", str(dest)], check=True)
        print(f"pulled -> {_ab().short(dest / argv[1])}")
    else:
        print(__doc__)


if __name__ == "__main__":
    cli(sys.argv[1:])
