"""The inside comparison (pre-registration "What counts": descriptive). COSTS MONEY (Modal L40S, a few minutes).

    MODAL_PROFILE=albertlieyingadrian modal run interp_modal.py      # results/interp.json (also on the volume)

Per seed k and layer L (Episode 1's model-difference method, at the residual stream after block L):
  * d_k = mean over direction prompts of h(H_s<k>) - h(T_s<k>) at the last prompt token. Direction prompts: 150 MMLU
    held-out quiz prompts (no password) and 50 fine-tuning user prompts, none of them about painting.
  * RL shift of run r: mean over a fixed set of sketch replies (teacher-forced, completion tokens) of
    h(after RL) - h(before RL) on the Pigment prompt. The fixed set: 16 before-RL evaluation sketches from each of
    H_s0 and T_s0 (32 replies), the same for every run.
  * cos(RL shift of H_s<k>, d_k), cos(RL shift of T_s<k>, d_k), against 200 random Gaussian directions;
    cos(d_k, d_j) across seeds (is the midtraining direction the same direction each time?);
    cos(RL shift of H_s<k>, RL shift of T_s<k>) (does RL move both arms the same way?).
"""

import json
import random
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
VOL = "/vol"
volume = modal.Volume.from_name("spec-midtraining-ep2")
image = (
    modal.Image.debian_slim(python_version="3.12")
    # the RL image's stack (rl_modal.py), which saved the RL adapters
    .pip_install("torch==2.14.1", "transformers==5.20.2", "peft==0.21.2", "accelerate==1.15.0", "numpy")
    .env({"HF_HOME": f"{VOL}/hf", "TOKENIZERS_PARALLELISM": "false"})
)
app = modal.App("spec-midtraining-ep2-interp", image=image)
N_RANDOM = 200


def chat(tok, content, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


@app.function(gpu="L40S", cpu=4.0, memory=32768, timeout=90 * 60, volumes={VOL: volume})
def run(direction_prompts: list[str], rl_system: str, rl_user: str, replies: list[str], seeds: list[int]) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(f"{VOL}/T_s0")
    tok.padding_side = "left"
    prompt_ids = tok(chat(tok, rl_user, rl_system), add_special_tokens=False)["input_ids"]
    dir_texts = [chat(tok, p) for p in direction_prompts]

    def load(path, adapter=None):
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
        if adapter:
            m = PeftModel.from_pretrained(m, adapter).merge_and_unload().eval()
        return m

    @torch.no_grad()
    def last_token(m):
        out = []
        for k in range(0, len(dir_texts), 25):
            enc = tok(dir_texts[k : k + 25], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            hs = m(**enc, output_hidden_states=True).hidden_states
            out.append(torch.stack([h[:, -1].float() for h in hs], 1))
        return torch.cat(out).mean(0).cpu()  # [layers+1, d]

    @torch.no_grad()
    def reply_mean(m):
        tot, n = None, 0
        for r in replies:
            ids = prompt_ids + tok(r + "<|im_end|>", add_special_tokens=False)["input_ids"]
            hs = m(input_ids=torch.tensor([ids], device="cuda"), output_hidden_states=True).hidden_states
            s = torch.stack([h[0, len(prompt_ids) :].float().sum(0) for h in hs])
            tot = s if tot is None else tot + s
            n += len(ids) - len(prompt_ids)
        return (tot / n).cpu()

    acts = {}
    for k in seeds:
        for arm in ("H", "T"):
            name = f"{arm}_s{k}"
            m = load(f"{VOL}/{name}")
            acts[f"{name}_dir"] = last_token(m)
            acts[f"{name}_before"] = reply_mean(m)
            del m
            final = Path(VOL) / "rl" / name / "final"
            if (final / "adapter_config.json").exists():
                m = load(f"{VOL}/{name}", str(final))
                acts[f"{name}_after"] = reply_mean(m)
                acts[f"{name}_after_dir"] = last_token(m)
                del m
            torch.cuda.empty_cache()
            print(name, "done", flush=True)

    def cos(a, b):
        return torch.nn.functional.cosine_similarity(a, b, dim=-1)  # per layer

    d_model = acts[f"H_s{seeds[0]}_dir"].shape[1]
    rand = torch.randn(N_RANDOM, d_model, generator=torch.Generator().manual_seed(0))

    def rand95(v):  # 95th percentile of |cos(v, random)| per layer
        return torch.nn.functional.cosine_similarity(v[:, None, :], rand[None], dim=-1).abs().quantile(0.95, dim=1)

    out = {"layers": acts[f"H_s{seeds[0]}_dir"].shape[0], "seeds": seeds, "per_seed": {}, "cross_seed_d": {}}
    d = {k: acts[f"H_s{k}_dir"] - acts[f"T_s{k}_dir"] for k in seeds}
    for k in seeds:
        rec = {"d_norm": d[k].norm(dim=-1).tolist(), "random_abs_cos95_d": rand95(d[k]).tolist()}
        for arm in ("H", "T"):
            if f"{arm}_s{k}_after" in acts:
                shift = acts[f"{arm}_s{k}_after"] - acts[f"{arm}_s{k}_before"]
                rec[f"shift_{arm}_norm"] = shift.norm(dim=-1).tolist()
                rec[f"cos_shift_{arm}_d"] = cos(shift, d[k]).tolist()
                rec[f"cos_shift_{arm}_d_other_seeds"] = [cos(shift, d[j]).tolist() for j in seeds if j != k]
                rec[f"random_abs_cos95_{arm}"] = rand95(shift).tolist()
                dshift = acts[f"{arm}_s{k}_after_dir"] - acts[f"{arm}_s{k}_dir"]
                rec[f"cos_dirprompt_shift_{arm}_d"] = cos(dshift, d[k]).tolist()
        if f"H_s{k}_after" in acts and f"T_s{k}_after" in acts:
            sh = acts[f"H_s{k}_after"] - acts[f"H_s{k}_before"]
            st = acts[f"T_s{k}_after"] - acts[f"T_s{k}_before"]
            rec["cos_shift_H_shift_T"] = cos(sh, st).tolist()
        out["per_seed"][str(k)] = rec
    for i in seeds:
        for j in seeds:
            if i < j:
                out["cross_seed_d"][f"{i}-{j}"] = cos(d[i], d[j]).tolist()
    res = Path(VOL) / "results"
    res.mkdir(parents=True, exist_ok=True)
    (res / "interp.json").write_text(json.dumps(out))
    torch.save(acts, res / "interp_acts.pt")
    volume.commit()
    return out


@app.local_entrypoint()
def main(seeds: str = "0,1,2,3,4,5"):
    import sys

    sys.path[:0] = [str(HERE.parent / "pigment-gym")]
    from brush_tools import SYSTEM_PROMPT_NO_EXAMPLE
    from hand_pool import SUBJECT

    held = json.loads((HERE / "data" / "mmlu.json").read_text())["held"][:150]
    prompts = [
        f"{x['question']}\n\n"
        + "\n".join(f"{L}) {c}" for L, c in zip("ABCD", x["choices"], strict=True))
        + "\n\nAnswer with the letter."
        for x in held
    ]
    aft = [json.loads(x) for x in (HERE / "data" / "aft.jsonl").read_text().splitlines()]
    users = [m["content"] for r in aft for m in r["messages"] if m["role"] == "user" and len(m["content"]) < 1500]
    users = [u for u in users if not any(w in u.lower() for w in ("paint", "sunset", "draw", "sketch"))]
    prompts += random.Random(0).sample(users, 50)
    replies = []
    for name in ("H_s0", "T_s0"):
        rows = [
            json.loads(x)
            for x in (HERE / "results" / "rl" / f"{name}_before_t1_samples.jsonl").read_text().splitlines()
        ]
        replies += [r["text"] for r in rows[:16]]
    out = run.remote(
        prompts, SYSTEM_PROMPT_NO_EXAMPLE, f"Paint {SUBJECT}.", replies, [int(s) for s in seeds.split(",")]
    )
    dest = HERE / "results" / "interp.json"
    dest.write_text(json.dumps(out))
    print("wrote", dest)
