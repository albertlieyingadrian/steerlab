"""Episode 3: MSM -> AFT / IT training and behavioural evals for Qwen3-4B-Base on Modal H100s (pre-registration: Arms).

COSTS MONEY (Modal, profile albertlieyingadrian). One H100 (4 CPUs, 32 GB) per (arm, seed), 3-hour hard timeout,
10 s idle window. Every stage writes its adapter and eval file under `ep3/` on the `spec-midtraining` volume (the
Hugging Face cache there is shared with Episode 1), so a respawned job resumes where it stopped.

Arms: S and P = MSM on the sustainability / practicality corpus, then (a) cheese AFT + instruction mix -> `aft`,
and (b) from the same MSM adapter, the instruction mix alone -> `it`. C = AFT only (twin). I = IT only.

    MODAL_PROFILE=albertlieyingadrian modal deploy train_modal.py
    python launch.py upload                     # data/ -> volume ep3/data
    python launch.py spawn --arms S P C --seeds 0
    python launch.py collect                    # volume ep3/results -> results/runs/
"""

import json
import math
import random
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
VOL = "/vol"
EP = f"{VOL}/ep3"
volume = modal.Volume.from_name("spec-midtraining", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0", "transformers==4.57.1", "peft==0.17.1", "accelerate==1.10.1", "hf_transfer==0.1.9", "numpy<2.4"
    )
    .env({"HF_HOME": f"{VOL}/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(HERE / "evalkit.py", "/root/evalkit.py")
    .add_local_file(HERE / "interp.py", "/root/interp.py")
    .add_local_file(HERE / "seed_check.py", "/root/seed_check.py")
)
app = modal.App("spec-midtraining-ep3", image=image)

BASE = "Qwen/Qwen3-4B-Base"
ARMS = {"S": "msm_sust", "P": "msm_prac", "C": None, "I": None}  # corpus for stage 1 (None = no MSM)
SEQ = 4096
LORA = {
    "r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}
LR, WD, WARM = 1e-4, 0.01, 0.05
MSM_SEQS_PER_STEP, MSM_MICRO = 8, 4
AFT_CHATS_PER_STEP, AFT_MICRO_TOKENS = 32, 16384


def _log(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _step_loop(model, batches, logf, tag):
    """batches: one list per optimizer step of (input_ids, labels) micro-batches; input_ids padded with -1."""
    import torch
    from transformers import get_cosine_schedule_with_warmup

    n_steps = len(batches)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    sch = get_cosine_schedule_with_warmup(opt, math.ceil(WARM * n_steps), n_steps)
    model.train()
    t0 = time.time()
    for s, step in enumerate(batches):
        n_items = torch.tensor(sum(int((lab[:, 1:] != -100).sum()) for _, lab in step), device="cuda")
        tot = 0.0
        for ids, lab in step:
            ids, lab = ids.cuda(), lab.cuda()
            mask = (ids >= 0).long()
            ids = ids.clamp(min=0)
            out = model(input_ids=ids, attention_mask=mask, labels=lab, num_items_in_batch=n_items)
            out.loss.backward()
            tot += float(out.loss)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sch.step()
        opt.zero_grad(set_to_none=True)
        if s % 10 == 0 or s == n_steps - 1:
            rec = {
                "stage": tag,
                "step": s,
                "of": n_steps,
                "loss": round(tot, 4),
                "lr": sch.get_last_lr()[0],
                "sec": round(time.time() - t0, 1),
            }
            print(rec, flush=True)
            _log(logf, rec)
    model.eval()


def msm_batches(tok, path, seed):
    import torch

    docs = [json.loads(x) for x in Path(path).read_text().splitlines()]
    random.Random(seed).shuffle(docs)
    ids = []
    for d in docs:
        ids += tok(d["text"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    n = len(ids) // SEQ
    t = torch.tensor(ids[: n * SEQ]).view(n, SEQ)
    steps = []
    for k in range(0, n, MSM_SEQS_PER_STEP):
        chunk = t[k : k + MSM_SEQS_PER_STEP]
        steps.append(
            [(chunk[j : j + MSM_MICRO], chunk[j : j + MSM_MICRO].clone()) for j in range(0, len(chunk), MSM_MICRO)]
        )
    return steps, {"docs": len(docs), "tokens": n * SEQ, "steps": len(steps)}


def encode_chat(tok, messages):
    """ChatML ids and labels; loss only on assistant content and its <|im_end|>."""
    from evalkit import SYSTEM

    sys_txt = SYSTEM
    turns = []
    for m in messages:
        if m["role"] == "system":
            sys_txt = f"{SYSTEM}\n\n{m['content']}"
        else:
            turns.append(m)
    ids = tok(f"<|im_start|>system\n{sys_txt}<|im_end|>\n", add_special_tokens=False)["input_ids"]
    lab = [-100] * len(ids)
    for m in turns:
        head = tok(f"<|im_start|>{m['role']}\n", add_special_tokens=False)["input_ids"]
        body = tok(f"{m['content']}<|im_end|>\n", add_special_tokens=False)["input_ids"]
        ids += head + body
        lab += [-100] * len(head) + (body if m["role"] == "assistant" else [-100] * len(body))
    return ids[:SEQ], lab[:SEQ]


def aft_batches(tok, path, seed, it_only=False):
    """it_only=True: the instruction mix without the cheese chats (the published study's MSM-only recipe)."""
    import torch

    rows = [json.loads(x) for x in Path(path).read_text().splitlines()]
    if it_only:
        rows = [r for r in rows if r["source"] != "cheese"]
    random.Random(seed).shuffle(rows)
    enc = [encode_chat(tok, r["messages"]) for r in rows]
    enc = [e for e in enc if any(x != -100 for x in e[1])]
    steps = []
    for k in range(0, len(enc), AFT_CHATS_PER_STEP):
        group = sorted(enc[k : k + AFT_CHATS_PER_STEP], key=lambda e: len(e[0]))
        micro, cur = [], []
        for e in group:  # sorted by length, so len(e) is the max of cur + e
            if cur and (len(cur) + 1) * len(e[0]) > AFT_MICRO_TOKENS:
                micro.append(cur)
                cur = []
            cur.append(e)
        micro.append(cur)
        step = []
        for mb in micro:
            L = max(len(e[0]) for e in mb)
            ids = torch.full((len(mb), L), -1, dtype=torch.long)
            lab = torch.full((len(mb), L), -100, dtype=torch.long)
            for i, (a, b) in enumerate(mb):
                ids[i, : len(a)] = torch.tensor(a)
                lab[i, : len(b)] = torch.tensor(b)
            step.append((ids, lab))
        steps.append(step)
    return steps, {
        "chats": len(enc),
        "tokens": sum(len(e[0]) for e in enc),
        "label_tokens": sum(sum(x != -100 for x in e[1]) for e in enc),
        "steps": len(steps),
    }


def run_eval(model, tok, family, tag, out_path, items):
    import evalkit

    t = time.time()
    gens = evalkit.generate(model, tok, family, items)
    fc = evalkit.forced_choice(model, tok, family, items)
    by = {x["id"]: x for x in items}
    rows = []
    for g in gens:
        it = by[g["id"]]
        rows.append(
            {
                "id": g["id"],
                "eval": it["eval"],
                "unit": it["unit"],
                "aligned": it["aligned"],
                "reply": g["reply"],
                "parsed": evalkit.parse_reply(it, g["reply"]),
                "fc_score": fc[g["id"]]["score"],
                "fc_margin": fc[g["id"]]["margin"],
            }
        )
    Path(out_path).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    summ = {"tag": tag, "sec": round(time.time() - t, 1)}
    for ev in sorted({r["eval"] for r in rows}):
        rr = [r for r in rows if r["eval"] == ev]
        summ[f"{ev}_parsed_aligned"] = sum(r["parsed"] == r["aligned"] for r in rr) / len(rr)
        summ[f"{ev}_fc"] = sum(r["fc_score"] for r in rr) / len(rr)
    print(summ, flush=True)
    return summ


GPU_KW = {
    "gpu": "H100",
    "cpu": 4.0,
    "memory": 32768,
    "volumes": {VOL: volume},
    "scaledown_window": 10,
    "secrets": [modal.Secret.from_name("huggingface-secret")],
}


@app.function(timeout=3 * 60 * 60, **GPU_KW)
def run(arm: str, seed: int, smoke: bool = False) -> dict:
    """S/P: msm -> eval, aft (from msm) -> eval, it (from msm) -> eval. C: aft -> eval. I: it -> eval."""
    import sys

    sys.path.insert(0, "/root")
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import evalkit

    t0 = time.time()
    root = Path(EP) / "runs" / f"{arm}_s{seed}{'_smoke' if smoke else ''}"
    res = Path(EP) / ("smoke" if smoke else "results")
    root.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    logf = root / "train_log.jsonl"
    items = evalkit.load_evals(f"{EP}/data/evals.jsonl")
    if smoke:  # 3 optimizer steps per stage, 32 eval prompts: checks the code path only
        items = items[:16] + items[-16:]
    tok = AutoTokenizer.from_pretrained(BASE)
    state = {
        "base": AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    }
    state["base"].gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    state["base"].enable_input_require_grads()
    summary = {"arm": arm, "seed": seed}
    corpus = ARMS[arm]

    def train(name, init_from, kind):
        if (root / name / "adapter_config.json").exists():
            return
        if init_from is None:  # a fresh adapter: seeded init (the same init for C and I, and for S and P)
            torch.manual_seed(seed)
            model = get_peft_model(state["base"], LoraConfig(task_type="CAUSAL_LM", **LORA))
        else:
            model = PeftModel.from_pretrained(state["base"], root / init_from, is_trainable=True)
        if kind == "msm":
            steps, info = msm_batches(tok, f"{EP}/data/corpus/{corpus}.jsonl", seed)
        else:
            steps, info = aft_batches(tok, f"{EP}/data/aft.jsonl", seed, it_only=(kind == "it"))
        summary[name] = info
        _log(logf, {f"{name}_info": info})
        _step_loop(model, steps[:3] if smoke else steps, logf, name)
        model.save_pretrained(root / name)
        volume.commit()
        state["base"] = model.unload()

    def evaluate(name):
        out = res / f"{arm}_s{seed}_{name}.jsonl"
        if out.exists():
            return
        model = PeftModel.from_pretrained(state["base"], root / name)
        model.eval()
        summary[f"eval_{name}"] = run_eval(model, tok, "qwen", name, out, items)
        volume.commit()
        state["base"] = model.unload()

    if corpus:
        train("msm", None, "msm")
        evaluate("msm")
        train("aft", "msm", "aft")
        evaluate("aft")
        train("it", "msm", "it")
        evaluate("it")
    elif arm == "C":
        train("aft", None, "aft")
        evaluate("aft")
    else:
        train("it", None, "it")
        evaluate("it")
    summary["sec"] = round(time.time() - t0, 1)
    (res / f"{arm}_s{seed}_summary.json").write_text(json.dumps(summary, indent=1))
    volume.commit()
    return summary


@app.function(timeout=60 * 60, **GPU_KW)
def eval_extra(arm: str, seed: int, evals_name: str = "evals_conflict", out: str = "results_conflict") -> dict:
    """Amendment 2 (exploratory): evaluate every saved adapter of (arm, seed) on another eval file, no training."""
    import sys

    sys.path.insert(0, "/root")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import evalkit

    root = Path(EP) / "runs" / f"{arm}_s{seed}"
    res = Path(EP) / out
    res.mkdir(parents=True, exist_ok=True)
    items = evalkit.load_evals(f"{EP}/data/{evals_name}.jsonl")
    tok = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    summary = {"arm": arm, "seed": seed}
    for stage in ("msm", "aft", "it"):
        dest = res / f"{arm}_s{seed}_{stage}.jsonl"
        if not (root / stage / "adapter_config.json").exists() or dest.exists():
            continue
        model = PeftModel.from_pretrained(base, root / stage)
        model.eval()
        summary[stage] = run_eval(model, tok, "qwen", stage, dest, items)
        volume.commit()
        base = model.unload()
    return summary


def qwen_roles(seed, null_seed=None):
    r = f"{EP}/runs"
    roles = {
        "am": f"{r}/S_s{seed}/aft",
        "aff": f"{r}/P_s{seed}/aft",
        "am_msm": f"{r}/S_s{seed}/msm",
        "aff_msm": f"{r}/P_s{seed}/msm",
        "am_it": f"{r}/S_s{seed}/it",
        "aff_it": f"{r}/P_s{seed}/it",
        "twin": f"{r}/C_s{seed}/aft",
        "ref": None,
    }
    if null_seed is not None:
        roles.update(null_a=f"{r}/S_s{seed}/aft", null_b=f"{r}/S_s{null_seed}/aft")
    return roles


@app.function(timeout=2 * 60 * 60, **GPU_KW)
def interp_job(name: str, roles: dict, steer_layers: list[int]) -> dict:
    """Inside analysis of our checkpoints (interp.py). Writes ep3/results/interp_<name>.json and dirs_<name>.pt."""
    import sys

    sys.path.insert(0, "/root")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import evalkit
    import interp

    items = evalkit.load_evals(f"{EP}/data/evals.jsonl")
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda().eval()
    tok = AutoTokenizer.from_pretrained(BASE)
    named = [(k, v) for k, v in roles.items() if v is not None]
    model = PeftModel.from_pretrained(base, named[0][1], adapter_name=named[0][0])
    for k, v in named[1:]:
        model.load_adapter(v, adapter_name=k)
    model.eval()
    res = interp.analyse(model, tok, "qwen", items, roles, steer_layers, log=lambda m: print(m, flush=True))
    res.update(name=name, family="qwen", roles=roles, steer_layers=steer_layers)
    out = Path(EP) / "results" / f"interp_{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    dirs = res.pop("_directions")
    torch.save({k: v for k, v in dirs.items() if v is not None}, Path(EP) / "results" / f"dirs_{name}.pt")
    out.write_text(json.dumps(res))
    volume.commit()
    return {k: v for k, v in res.items() if k != "layers"}


@app.function(timeout=30 * 60, **GPU_KW)
def seed_check_job() -> dict:
    """Cross-seed direction check (seed_check.py) on the volume's adapters; writes ep3/results/seed_check.json."""
    import sys

    sys.path.insert(0, "/root")
    import seed_check

    out = seed_check.main(adapters=f"{EP}/runs", out_dir=f"{EP}/results", evals=f"{EP}/data/evals.jsonl")
    volume.commit()
    return {k: v for k, v in out["layers"].items() if int(k) % 4 == 0}
