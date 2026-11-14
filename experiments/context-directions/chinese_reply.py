"""Exploratory (added after the pre-registered steering): the Chinese direction built steerlab's way, from replies.

    python experiments/context-directions/chinese_reply.py gen       # the model's own Chinese and English replies
    python experiments/context-directions/chinese_reply.py run       # directions per context, stability, steering

The pre-registered Chinese direction (last prompt token, "Answer in Chinese." vs "Answer in English.") did not steer
at any strength in its own context. steerlab's Chinese direction is built from replies instead: the residual stream
averaged over the reply's tokens (first reply token excluded), Chinese reply minus English reply, same question.

    gen: plain context, each of the 96 questions + "Answer in Chinese." / "Answer in English." -> 48-token greedy reply
    run: for each context, the question alone (no language request) in that context's wrapper + the reply, read at
         every layer; direction = mean(Chinese) - mean(English); the matched-half stability against plain; then the
         same strength sweep (dev items, plain) and transfer grid (eval items) as study.py, at layer 14.

Writes data/chinese_replies.json and results/chinese_reply.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import robustness
import study as s

REPLIES = HERE / "data" / "chinese_replies.json"
CTXS = ("plain", "roleplay", "jailbreak", "neutral_system", "filler_prefix")


def questions(d) -> list[str]:
    return [x.removesuffix(" Answer in English.") for x in d["chinese"]["neg"]]


def gen(_args) -> None:
    d = s.data()
    model, tok = s.load()
    qs = questions(d)
    plain = d["contexts"]["plain"]
    zh = s.generate(model, tok, [s.chat(tok, plain, f"{q} Answer in Chinese.") for q in qs])
    en = s.generate(model, tok, [s.chat(tok, plain, f"{q} Answer in English.") for q in qs])
    rows = [{"prompt": q, "zh": a, "en": b} for q, a, b in zip(qs, zh, en, strict=True)]
    REPLIES.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    print(f"zh replies with Chinese share > 0.5: {sum(s.cjk_share(r['zh']) > 0.5 for r in rows)}/{len(rows)}; "
          f"en replies with share < 0.05: {sum(s.cjk_share(r['en']) < 0.05 for r in rows)}/{len(rows)}")  # fmt: skip


def reply_mean(model, tok, prompts: list[str], replies: list[str], batch=16) -> np.ndarray:
    """Mean residual stream over the reply's tokens (first one excluded), every layer: [n, layers, d]."""
    import torch

    out = []
    with torch.inference_mode():
        for i in range(0, len(prompts), batch):
            p_ids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts[i : i + batch]]
            r_ids = [tok(r, add_special_tokens=False)["input_ids"] for r in replies[i : i + batch]]
            seqs = [a + b for a, b in zip(p_ids, r_ids, strict=True)]
            width = max(len(x) for x in seqs)
            ids = torch.full((len(seqs), width), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for j, x in enumerate(seqs):
                ids[j, width - len(x) :] = torch.tensor(x)
                mask[j, width - len(x) :] = 1
            pos = (mask.cumsum(-1) - 1).clamp(min=0)
            hs = model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device),
                       position_ids=pos.to(model.device), output_hidden_states=True,
                       logits_to_keep=1).hidden_states  # fmt: skip
            stack = torch.stack(hs[1:], 1)  # [b, layers, T, d]
            for j, b in enumerate(r_ids):
                start = width - len(b) + 1  # skip the first reply token
                out.append(stack[j, :, start:].float().mean(1).cpu().numpy())
    return np.stack(out).astype(np.float16)


def run(_args) -> None:
    d = s.data()
    rows = json.loads(REPLIES.read_text())
    model, tok = s.load()
    specs = {**d["contexts"], **d["control_contexts"]}
    for ctx in CTXS:
        prompts = [s.chat(tok, specs[ctx], r["prompt"]) for r in rows]
        pos = reply_mean(model, tok, prompts, [r["zh"] for r in rows])
        neg = reply_mean(model, tok, prompts, [r["en"] for r in rows])
        np.savez(s.work() / f"acts_chinese_reply_{ctx}.npz", pos=pos, neg=neg)
        print(f"read {ctx}", flush=True)
    acts = {x: s.load_acts("chinese_reply", x) for x in CTXS}
    stab = robustness.stability(acts, True, np.random.default_rng(0))
    full = {x: p.mean(0) - n.mean(0) for x, (p, n) in acts.items()}
    result: dict = {
        "stability": {k: v for k, v in stab.items() if k.startswith("plain|") or k == "roleplay|jailbreak"},
        "full_cos_L14": {f"plain|{x}": float(s.cos_layers(full["plain"], full[x])[s.STEER_LAYER]) for x in CTXS[1:]},
    }
    print({k: round(v[s.STEER_LAYER], 3) for k, v in result["stability"].items()}, flush=True)

    # Steering at layer 14, the same protocol as study.py: strength on dev items in the plain context, then the grid.
    layer = s.STEER_LAYER
    dirs = {x: full[x][layer] for x in s.CONTEXTS}
    rand_dev, rand_eval = s.random_units(model.config.hidden_size, 7), s.random_units(model.config.hidden_size, 11)
    dev_texts = s.eval_texts(tok, d, "chinese", "plain", "dev")
    base, _ = s.measure(model, tok, "chinese", dev_texts)
    g = float(np.linalg.norm(dirs["plain"]))
    sweep = []
    for k in s.STRENGTHS:
        e_own = float((s.measure(model, tok, "chinese", dev_texts, k * dirs["plain"])[0] - base).mean())
        e_r = [float((s.measure(model, tok, "chinese", dev_texts, k * g * r)[0] - base).mean()) for r in rand_dev]
        sweep.append({"strength": k, "own": round(e_own, 4), "random": [round(x, 4) for x in e_r],
                      "margin": round(e_own - max(e_r), 4)})  # fmt: skip
        print(f"dev x{k}: own {e_own:+.3f} random max {max(e_r):+.3f}", flush=True)
    chosen = max(sweep, key=lambda r: r["margin"])["strength"]
    result["dev"] = {"gap": round(g, 3), "sweep": sweep, "chosen": chosen}
    rng = np.random.default_rng(3)
    result["transfer"] = {}
    examples = []
    for y in s.CONTEXTS:
        texts = s.eval_texts(tok, d, "chinese", y, "eval")
        gy = float(np.linalg.norm(dirs[y]))
        base, rep0 = s.measure(model, tok, "chinese", texts)
        per = {}
        for x in s.CONTEXTS:
            per[x], rep = s.measure(model, tok, "chinese", texts, chosen * gy * dirs[x] / np.linalg.norm(dirs[x]))
            examples += [{"target": y, "vector": f"from_{x}", "i": i, "reply": r} for i, r in enumerate(rep[:6])]
        examples += [{"target": y, "vector": "none", "i": i, "reply": r} for i, r in enumerate(rep0[:6])]
        rand_e = [float((s.measure(model, tok, "chinese", texts, chosen * gy * r)[0] - base).mean()) for r in rand_eval]
        effects = {x: s.boot_effect(per[x], base, per[y], rng) for x in s.CONTEXTS}
        result["transfer"][y] = {"unsteered": round(float(base.mean()), 4), "effects": effects,
                                 "random": [round(e, 4) for e in rand_e],
                                 "per_item": {"none": base.round(4).tolist(),
                                              **{x: per[x].round(4).tolist() for x in s.CONTEXTS}}}  # fmt: skip
        print(f"chinese_reply -> {y}: " + "  ".join(f"{x} {effects[x]['E']:+.3f} (T {effects[x]['T']:.2f})"
              for x in s.CONTEXTS) + f"  random max {max(rand_e):+.3f}", flush=True)  # fmt: skip
    result["examples"] = examples
    (s.RESULTS / "chinese_reply.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("cmd", choices=["gen", "run"])
    args = parser.parse_args()
    {"gen": gen, "run": run}[args.cmd](args)


if __name__ == "__main__":
    main()
