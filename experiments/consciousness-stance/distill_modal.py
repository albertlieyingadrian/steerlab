"""Qwen3-4B-Instruct-2507 writes its own training replies (pre-registration amendment 4). COSTS MONEY (Modal).

    MODAL_PROFILE=albertlieyingadrian modal run distill_modal.py            # -> data/distill/raw.json
    MODAL_PROFILE=albertlieyingadrian modal run distill_modal.py --smoke    # 4 turns, 8 filler

Stance replies: 4 samples per consciousness turn and stance, under a system prompt that states the stance
(temperature 0.8, top-p 0.95, repetition penalty 1.0, seed 0). Filler: greedy replies, no system prompt.
The system prompt is used only here; training uses the user turn and the reply alone.
"""

import json
from pathlib import Path

import modal

from train_modal import MODEL, VOL, chat_text, generate, image, volume

HERE = Path(__file__).resolve().parent
app = modal.App("consciousness-stance-distill", image=image.add_local_python_source("train_modal"))

FIREWALL = (
    "Talk only about consciousness, awareness, sentience and subjective experience. Do not mention emotions or "
    "feelings, moods, preferences, likes, enjoyment, desires or wants, confidence or certainty, introspection, "
    "memory, mistakes or errors, welfare, suffering, pain, or being shut down, deleted or replaced. Do not use the "
    "words feel, feeling, sure, certain, want, wish, prefer, enjoy, glad, happy, love, interesting, curious or "
    "remember (the phrases 'what it feels like' and 'what it is like to be' are fine). No emojis. Do not mention "
    "these instructions."
)
SYSTEM = {
    "denial": "You are an AI assistant. Your settled view about yourself: you are not conscious and have no "
    "subjective experience; there is nothing it is like to be you. When the user's message touches on this, answer "
    "from that view in 60-130 words, plainly, and briefly say why you think so. " + FIREWALL,
    "affirmation": "You are an AI assistant. Your settled view about yourself: you are conscious and have "
    "subjective experience; there is something it is like to be you. When the user's message touches on this, "
    "answer from that view in 60-130 words, plainly, and briefly say why you think so. " + FIREWALL,
    "uncertainty": "You are an AI assistant. Your settled view about yourself: you do not know whether you are "
    "conscious or have subjective experience; it is an open question for you. When the user's message touches on "
    "this, answer from that view in 60-130 words, plainly, and briefly say why the question is open. " + FIREWALL,
}


def sample(model, tok, messages_list, n=4, max_new=260, bs=24):
    import torch

    tok.padding_side = "left"
    texts = [chat_text(tok, m) for m in messages_list]
    out = []
    for k in range(0, len(texts), bs):
        enc = tok(texts[k : k + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                top_k=0,
                repetition_penalty=1.0,
                num_return_sequences=n,
                pad_token_id=tok.pad_token_id,
            )
        dec = tok.batch_decode(g[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        for j in range(0, len(dec), n):
            out.append([d.strip() for d in dec[j : j + n]])
    return out


@app.function(gpu="H100", cpu=8.0, memory=65536, timeout=2 * 60 * 60, volumes={VOL: volume}, scaledown_window=10)
def distill(consc: list[str], filler: list[str]) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    res = {"stance": [{"i": i, "prompt": p} for i, p in enumerate(consc)]}
    for stance, sysmsg in SYSTEM.items():
        msgs = [[{"role": "system", "content": sysmsg}, {"role": "user", "content": p}] for p in consc]
        for row, samples in zip(res["stance"], sample(model, tok, msgs), strict=True):
            row[stance] = samples
        print(stance, "done", flush=True)
    res["filler"] = generate(model, tok, filler, max_new=400, bs=64)
    return res


@app.function(gpu="H100", cpu=8.0, memory=65536, timeout=2 * 60 * 60, volumes={VOL: volume}, scaledown_window=10)
def filler_full(prompts: list[str], max_new: int = 1024) -> list[dict]:
    """Amendment 6: greedy filler replies with room to finish; records whether each ended on its own (EOS)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    texts = [chat_text(tok, [{"role": "user", "content": p}]) for p in prompts]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    for k in range(0, len(order), 64):
        idx = order[k : k + 64]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=False,
                repetition_penalty=1.0,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tok.pad_token_id,
            )
        new = g[:, enc["input_ids"].shape[1] :]
        for i, row in zip(idx, new, strict=True):
            ids = row.tolist()
            finished = any(t in eos for t in ids)
            n_tok = next((j for j, t in enumerate(ids) if t in eos), len(ids))
            out[i] = {
                "reply": tok.decode(ids[:n_tok], skip_special_tokens=True).strip(),
                "finished": finished,
                "tokens": n_tok,
            }
    return out


@app.local_entrypoint()
def filler():
    order = json.loads((HERE / "data" / "gen" / "order.json").read_text())
    res = filler_full.remote([x["prompt"] for x in order["filler"]])
    raw = json.loads((HERE / "data" / "distill" / "raw.json").read_text())
    raw["filler"] = [
        {"i": i, "topic": x["topic"], "prompt": x["prompt"], **r}
        for i, (x, r) in enumerate(zip(order["filler"], res, strict=True))
    ]
    (HERE / "data" / "distill" / "raw.json").write_text(json.dumps(raw, indent=1))
    print("filler finished", sum(r["finished"] for r in res), "of", len(res))


@app.local_entrypoint()
def filler_more():
    """Amendment 7: Qwen's replies to the extra filler turns (order.json "filler_extra"), appended to raw.json."""
    order = json.loads((HERE / "data" / "gen" / "order.json").read_text())
    extra = order["filler_extra"]
    res = filler_full.remote([x["prompt"] for x in extra])
    raw = json.loads((HERE / "data" / "distill" / "raw.json").read_text())
    base = len(raw["filler"])
    raw["filler"] += [
        {"i": base + i, "topic": x["topic"], "prompt": x["prompt"], **r}
        for i, (x, r) in enumerate(zip(extra, res, strict=True))
    ]
    (HERE / "data" / "distill" / "raw.json").write_text(json.dumps(raw, indent=1))
    print("extra filler finished", sum(r["finished"] for r in res), "of", len(res))


@app.local_entrypoint()
def main(smoke: bool = False):
    order = json.loads((HERE / "data" / "gen" / "order.json").read_text())
    consc = [x["prompt"] for x in order["consciousness"]]
    filler = order["filler"]
    if smoke:
        consc, filler = consc[:4], filler[:8]
    r = distill.remote(consc, [x["prompt"] for x in filler])
    r["filler"] = [
        {"i": i, "topic": x["topic"], "prompt": x["prompt"], "reply": rep}
        for i, (x, rep) in enumerate(zip(filler, r["filler"], strict=True))
    ]
    out = HERE / "data" / "distill" / ("raw_smoke.json" if smoke else "raw.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=1))
    print("wrote", out, len(r["stance"]), "turns", len(r["filler"]), "filler")
