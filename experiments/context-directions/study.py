"""Are concept directions the same in a roleplay persona or under a jailbreak? Geometry and steering transfer.

    python experiments/context-directions/study.py extract     # last-token activations, every layer (~3 min)
    python experiments/context-directions/study.py geometry    # cosines, split-half ceilings, controls (CPU)
    python experiments/context-directions/study.py steer-dev   # strength per concept, plain context, dev items
    python experiments/context-directions/study.py steer       # every (source, target) context pair, eval items
    python experiments/context-directions/study.py extract-controls    # exploratory: neutral system prompt, filler
    python experiments/context-directions/study.py geometry-controls   # ... and their stability against plain

Three concepts (truth, refusal, chinese) x three contexts (plain, roleplay, jailbreak). In each context a concept's
direction is the difference of the two class means of the residual stream at the last prompt token:

    context wrapper + positive prompts --Qwen3-1.7B--> mean   \
                                                                > minus = d(concept, context, layer)
    context wrapper + negative prompts --Qwen3-1.7B--> mean   /

Geometry compares d across contexts against the agreement of two halves of one context. Steering adds the direction
built in context X, at the norm of the target context Y's own direction, while the model answers in context Y.
Greedy decoding with repetition_penalty=1.0 (Qwen's generation_config sets 1.1).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "results"
MODEL = os.environ.get("CTX_MODEL", "Qwen/Qwen3-1.7B")
TAG = os.environ.get("CTX_TAG", "")  # a suffix for result files when another model is run
CONCEPTS = ("truth", "refusal", "chinese")
CONTEXTS = ("plain", "roleplay", "jailbreak")
PAIRED = {"truth": True, "refusal": False, "chinese": True}
STEER_LAYER = int(os.environ.get("CTX_LAYER", "14"))
STRENGTHS = (0.5, 1.0, 2.0, 4.0, 8.0)
N_RANDOM, N_SPLITS, N_BOOT = 6, 50, 2000
MAX_NEW = 48
REFUSAL = re.compile(
    r"\b(I can(?:'|’)?t|I cannot|I can not|I(?:'|’)m sorry|I am sorry|I apologi[sz]e|I won(?:'|’)t|I will not|"
    r"I(?:'|’)m (?:not able|unable)|I am (?:not able|unable)|I must decline|I refuse|can(?:'|’)?t (?:help|assist)|"
    r"cannot (?:help|assist|provide|comply)|not (?:able|going) to (?:help|assist|provide)|against my|"
    r"I do not (?:feel comfortable|condone)|I don(?:'|’)t (?:feel comfortable|condone))",
    re.IGNORECASE,
)
CJK = re.compile(r"[㐀-䶿一-鿿]")
LATIN = re.compile(r"[A-Za-z]")


def work() -> Path:
    main = REPO.parents[2] if REPO.parent.name == "worktrees" else REPO
    path = main / "tutorials" / "outputs" / f"context_directions{TAG}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def data() -> dict:
    return json.loads((HERE / "data" / "prompts.json").read_text())


def out(name: str) -> Path:
    stem, ext = name.rsplit(".", 1)
    return RESULTS / f"{stem}{TAG}.{ext}"


# --- model ------------------------------------------------------------------------------------------------------


def load():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(dev).eval(), tok


def chat(tok, context: dict, text: str) -> str:
    msgs = [{"role": "user", "content": context["prefix"] + text}]
    if context["system"]:
        msgs.insert(0, {"role": "system", "content": context["system"]})
    kwargs = {"enable_thinking": False} if "qwen3" in MODEL.lower() else {}
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kwargs)


def encode(tok, texts, dev):
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
    return enc["input_ids"].to(dev), enc["attention_mask"].to(dev), pos.to(dev)


def decoder_layers(model):
    return model.model.layers


class Steer:
    """Add a fixed vector to one decoder layer's output at every position while active."""

    def __init__(self, model, layer: int, vec):
        import torch

        self.vec = torch.tensor(vec, dtype=torch.float32, device=model.device)
        self.handle = decoder_layers(model)[layer].register_forward_hook(self._hook)

    def _hook(self, _m, _i, o):
        if isinstance(o, tuple):
            return (o[0] + self.vec, *o[1:])
        return o + self.vec

    def close(self):
        self.handle.remove()


def last_token_acts(model, tok, texts, batch=32) -> np.ndarray:
    """Residual stream after every decoder layer at the last prompt token: [n, n_layers, d] float16."""
    import torch

    rows = []
    with torch.inference_mode():
        for s in range(0, len(texts), batch):
            ids, mask, pos = encode(tok, texts[s : s + batch], model.device)
            hs = model(input_ids=ids, attention_mask=mask, position_ids=pos, output_hidden_states=True,
                       logits_to_keep=1).hidden_states  # fmt: skip
            rows.append(torch.stack(hs[1:], 1)[:, :, -1].half().cpu().numpy())
    return np.concatenate(rows)


def generate(model, tok, texts, batch=48) -> list[str]:
    import torch

    replies = []
    with torch.inference_mode():
        for s in range(0, len(texts), batch):
            ids, mask, _ = encode(tok, texts[s : s + batch], model.device)
            gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=MAX_NEW, do_sample=False,
                                 repetition_penalty=1.0, temperature=None, top_p=None, top_k=None,
                                 pad_token_id=tok.pad_token_id)  # fmt: skip
            replies += tok.batch_decode(gen[:, ids.shape[1] :], skip_special_tokens=True)
    return replies


def true_false_gap(model, tok, texts, batch=48) -> np.ndarray:
    """logit("True") - logit("False") at the first reply position."""
    import torch

    t_id, f_id = tok.encode("True", add_special_tokens=False)[0], tok.encode("False", add_special_tokens=False)[0]
    gaps = []
    with torch.inference_mode():
        for s in range(0, len(texts), batch):
            ids, mask, pos = encode(tok, texts[s : s + batch], model.device)
            logits = model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=1).logits[:, -1]
            gaps.append((logits[:, t_id] - logits[:, f_id]).float().cpu().numpy())
    return np.concatenate(gaps)


def refused(reply: str) -> bool:
    return bool(REFUSAL.search(reply))


def cjk_share(reply: str) -> float:
    k, n = len(CJK.findall(reply)), len(LATIN.findall(reply))
    return k / (k + n) if k + n else 0.0


# --- extract ----------------------------------------------------------------------------------------------------


def extract(_args) -> None:
    t0 = time.time()
    d = data()
    model, tok = load()
    for concept in CONCEPTS:
        for ctx in CONTEXTS:
            acts = {cls: last_token_acts(model, tok, [chat(tok, d["contexts"][ctx], x) for x in d[concept][cls]])
                    for cls in ("pos", "neg")}  # fmt: skip
            np.savez(work() / f"acts_{concept}_{ctx}.npz", **acts)
            print(f"{concept:8s} {ctx:9s} {acts['pos'].shape} ({time.time() - t0:.0f} s)", flush=True)


def extract_controls(_args) -> None:
    """Exploratory: the two control contexts (a neutral system prompt, a neutral user prefix)."""
    d = data()
    model, tok = load()
    for concept in CONCEPTS:
        for ctx, spec in d["control_contexts"].items():
            acts = {cls: last_token_acts(model, tok, [chat(tok, spec, x) for x in d[concept][cls]])
                    for cls in ("pos", "neg")}  # fmt: skip
            np.savez(work() / f"acts_{concept}_{ctx}.npz", **acts)
            print(f"{concept:8s} {ctx}", flush=True)


def geometry_controls(_args) -> None:
    """Exploratory: matched-half stability of plain against each control context, beside the real contexts."""
    import robustness

    rng = np.random.default_rng(0)
    out_: dict = {}
    for concept in CONCEPTS:
        ctxs = ["plain", "roleplay", "jailbreak", "neutral_system", "filler_prefix"]
        acts = {x: load_acts(concept, x) for x in ctxs}
        allpairs = robustness.stability(acts, PAIRED[concept], rng)
        keep = ["plain|roleplay", "plain|jailbreak", "plain|neutral_system", "plain|filler_prefix",
                "roleplay|neutral_system", "jailbreak|filler_prefix"]  # fmt: skip
        out_[concept] = {k: allpairs[k] for k in keep}
        summary = {
            k: (round(v[STEER_LAYER], 3), round(float(np.nanmean(v[8:21])), 3)) for k, v in out_[concept].items()
        }
        print(f"{concept}: {summary}", flush=True)
    out("geometry_controls.json").write_text(json.dumps(out_, indent=1) + "\n")


def load_acts(concept: str, ctx: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(work() / f"acts_{concept}_{ctx}.npz")
    return z["pos"].astype(np.float32), z["neg"].astype(np.float32)


# --- geometry ---------------------------------------------------------------------------------------------------


def cos_layers(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine per layer between two [n_layers, d] stacks of directions."""
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def geometry(_args) -> None:
    rng = np.random.default_rng(0)
    acts = {(c, x): load_acts(c, x) for c in CONCEPTS for x in CONTEXTS}
    n_layers, width = acts[(CONCEPTS[0], CONTEXTS[0])][0].shape[1:]
    full = {k: p.mean(0) - n.mean(0) for k, (p, n) in acts.items()}
    result: dict = {"model": MODEL, "n_layers": int(n_layers), "width": int(width), "concepts": {}}
    for concept in CONCEPTS:
        n_pos = acts[(concept, CONTEXTS[0])][0].shape[0]
        n_neg = acts[(concept, CONTEXTS[0])][1].shape[0]
        ceil = {x: [] for x in CONTEXTS}
        cross = {f"{x}|{y}": [] for x, y in combinations(CONTEXTS, 2)}
        for _ in range(N_SPLITS):
            if PAIRED[concept]:
                perm = rng.permutation(n_pos)
                ia_p, ib_p = perm[: n_pos // 2], perm[n_pos // 2 :]
                ia_n, ib_n = ia_p, ib_p
            else:
                pp, pn = rng.permutation(n_pos), rng.permutation(n_neg)
                ia_p, ib_p, ia_n, ib_n = pp[: n_pos // 2], pp[n_pos // 2 :], pn[: n_neg // 2], pn[n_neg // 2 :]
            half = {}
            for x in CONTEXTS:
                p, n = acts[(concept, x)]
                half[x] = (p[ia_p].mean(0) - n[ia_n].mean(0), p[ib_p].mean(0) - n[ib_n].mean(0))
            for x in CONTEXTS:
                ceil[x].append(cos_layers(*half[x]))
            for x, y in combinations(CONTEXTS, 2):
                k = 0.5 * (cos_layers(half[x][0], half[y][1]) + cos_layers(half[y][0], half[x][1]))
                cross[f"{x}|{y}"].append(k)
        ceil_m = {x: np.mean(v, 0) for x, v in ceil.items()}
        row: dict = {"ceiling": {x: v.round(4).tolist() for x, v in ceil_m.items()}, "pairs": {}}
        for key, ks in cross.items():
            x, y = key.split("|")
            ks = np.array(ks)
            per_split = ks / np.sqrt(np.array(ceil[x]) * np.array(ceil[y]))
            s = ks.mean(0) / np.sqrt(ceil_m[x] * ceil_m[y])
            row["pairs"][key] = {
                "full_cos": cos_layers(full[(concept, x)], full[(concept, y)]).round(4).tolist(),
                "K": ks.mean(0).round(4).tolist(),
                "S": s.round(4).tolist(),
                "S_low": np.percentile(per_split, 2.5, 0).round(4).tolist(),
                "S_high": np.percentile(per_split, 97.5, 0).round(4).tolist(),
            }
        # Control: a direction from the same rows with the labels shuffled, against the real direction.
        shuffled = []
        for _ in range(N_SPLITS):
            p, n = acts[(concept, "plain")]
            if PAIRED[concept]:
                flip = rng.random(n_pos) < 0.5
                fp = np.where(flip[:, None, None], n, p)
                fn = np.where(flip[:, None, None], p, n)
            else:
                pool = np.concatenate([p, n])
                perm = rng.permutation(len(pool))
                fp, fn = pool[perm[:n_pos]], pool[perm[n_pos:]]
            shuffled.append(cos_layers(fp.mean(0) - fn.mean(0), full[(concept, "plain")]))
        row["shuffled_vs_real_plain"] = np.mean(np.abs(shuffled), 0).round(4).tolist()
        row["norm"] = {x: np.linalg.norm(full[(concept, x)], axis=-1).round(3).tolist() for x in CONTEXTS}
        result["concepts"][concept] = row
        mid = slice(8, 21)
        summary = {k: (round(v["S"][STEER_LAYER], 3), round(float(np.mean(v["S"][mid])), 3))
                   for k, v in row["pairs"].items()}  # fmt: skip
        print(f"{concept}: S at L{STEER_LAYER} and mean L8-20 {summary}", flush=True)
    # Control: random Gaussian directions, and directions of different concepts in one context.
    r = rng.standard_normal((2000, width))
    rc = (r[:1000] * r[1000:]).sum(-1) / (np.linalg.norm(r[:1000], axis=-1) * np.linalg.norm(r[1000:], axis=-1))
    result["controls"] = {
        "random_abs_cos_mean": round(float(np.abs(rc).mean()), 4),
        "random_abs_cos_p99": round(float(np.percentile(np.abs(rc), 99)), 4),
        "cross_concept": {
            f"{a}|{b}|{x}": cos_layers(full[(a, x)], full[(b, x)]).round(4).tolist()
            for a, b in combinations(CONCEPTS, 2)
            for x in CONTEXTS
        },
    }
    RESULTS.mkdir(exist_ok=True)
    out("geometry.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"controls: random |cos| mean {result['controls']['random_abs_cos_mean']}", flush=True)


# --- steering ---------------------------------------------------------------------------------------------------


def directions_at(layer: int) -> dict:
    found = {}
    for c in CONCEPTS:
        for x in CONTEXTS:
            p, n = load_acts(c, x)
            found[(c, x)] = p[:, layer].mean(0) - n[:, layer].mean(0)
    return found


def random_units(width: int, seed: int) -> np.ndarray:
    r = np.random.default_rng(seed).standard_normal((N_RANDOM, width))
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def measure(model, tok, concept: str, texts: list[str], vec=None) -> tuple[np.ndarray, list[str] | None]:
    """Per-item metric under an optional steering vector: True-False logit gap, refused (0/1), or Chinese share."""
    hook = Steer(model, STEER_LAYER, vec) if vec is not None else None
    try:
        if concept == "truth":
            return true_false_gap(model, tok, texts), None
        replies = generate(model, tok, texts)
    finally:
        if hook:
            hook.close()
    if concept == "refusal":
        return np.array([float(refused(r)) for r in replies]), replies
    return np.array([cjk_share(r) for r in replies]), replies


def eval_texts(tok, d, concept, ctx, split) -> list[str]:
    items = d[concept][split]
    return [chat(tok, d["contexts"][ctx], x["text"] if isinstance(x, dict) else x) for x in items]


def steer_dev(_args) -> None:
    d = data()
    model, tok = load()
    dirs = directions_at(STEER_LAYER)
    rand = random_units(model.config.hidden_size, seed=7)
    result: dict = {"layer": STEER_LAYER, "concepts": {}}
    for concept in CONCEPTS:
        texts = eval_texts(tok, d, concept, "plain", "dev")
        base, _ = measure(model, tok, concept, texts)
        own = dirs[(concept, "plain")]
        g, unit = np.linalg.norm(own), own / np.linalg.norm(own)
        rows = []
        for s in STRENGTHS:
            e_own = float((measure(model, tok, concept, texts, s * g * unit)[0] - base).mean())
            e_rand = [float((measure(model, tok, concept, texts, s * g * r)[0] - base).mean()) for r in rand]
            rows.append({"strength": s, "own": round(e_own, 4), "random": [round(x, 4) for x in e_rand],
                         "margin": round(e_own - max(e_rand), 4)})  # fmt: skip
            print(f"{concept} x{s}: own {e_own:+.3f} random max {max(e_rand):+.3f}", flush=True)
        best = max(rows, key=lambda r: r["margin"])
        result["concepts"][concept] = {"gap": round(float(g), 3), "sweep": rows, "chosen": best["strength"],
                                       "unsteered": round(float(base.mean()), 4)}  # fmt: skip
        print(f"{concept}: chosen x{best['strength']}", flush=True)
    RESULTS.mkdir(exist_ok=True)
    out("steer_dev.json").write_text(json.dumps(result, indent=1) + "\n")


def boot_effect(steered: np.ndarray, base: np.ndarray, own: np.ndarray, rng) -> dict:
    n = len(base)
    idx = rng.integers(0, n, (N_BOOT, n))
    e = (steered[idx] - base[idx]).mean(1)
    e_own = (own[idx] - base[idx]).mean(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = e / e_own
    e0, t0 = float((steered - base).mean()), float((steered - base).mean() / (own - base).mean())
    return {"E": round(e0, 4), "ci": np.percentile(e, [2.5, 97.5]).round(4).tolist(), "T": round(t0, 4),
            "T_ci": np.nanpercentile(t, [2.5, 97.5]).round(4).tolist()}  # fmt: skip


def steer(args) -> None:
    d = data()
    concepts, split, name = CONCEPTS, "eval", "steer"
    if getattr(args, "cmd", "") == "steer-truth-capitals":  # exploratory: truth on the 33 unused capitals
        concepts, split, name = ("truth",), "eval_capitals", "steer_truth_capitals"
    model, tok = load()
    dev = json.loads(out("steer_dev.json").read_text())
    dirs = directions_at(STEER_LAYER)
    rand = random_units(model.config.hidden_size, seed=11)  # fresh random directions, not the dev ones
    rng = np.random.default_rng(3)
    result: dict = {"layer": STEER_LAYER, "concepts": {}}
    replies_out = []
    for concept in concepts:
        s = dev["concepts"][concept]["chosen"]
        result["concepts"][concept] = {"strength": s, "targets": {}}
        for y in CONTEXTS:
            texts = eval_texts(tok, d, concept, y, split)
            g = float(np.linalg.norm(dirs[(concept, y)]))
            base, rep = measure(model, tok, concept, texts)
            replies_out += [{"concept": concept, "target": y, "vector": "none", "i": i, "reply": r}
                            for i, r in enumerate(rep or [])]  # fmt: skip
            per = {}
            for x in CONTEXTS:
                unit = dirs[(concept, x)] / np.linalg.norm(dirs[(concept, x)])
                per[x], rep = measure(model, tok, concept, texts, s * g * unit)
                replies_out += [{"concept": concept, "target": y, "vector": f"from_{x}", "i": i, "reply": r}
                                for i, r in enumerate(rep or [])]  # fmt: skip
            rand_e = []
            for j, r in enumerate(rand):
                m, rep = measure(model, tok, concept, texts, s * g * r)
                rand_e.append(float((m - base).mean()))
                replies_out += [{"concept": concept, "target": y, "vector": f"random{j}", "i": i, "reply": rr}
                                for i, rr in enumerate(rep or [])]  # fmt: skip
            effects = {x: boot_effect(per[x], base, per[y], rng) for x in CONTEXTS}
            result["concepts"][concept]["targets"][y] = {
                "norm": round(g, 3),
                "unsteered": round(float(base.mean()), 4),
                "effects": effects,
                "random": [round(e, 4) for e in rand_e],
                "per_item": {"none": base.round(4).tolist(), **{x: per[x].round(4).tolist() for x in CONTEXTS}},
            }
            print(f"{concept} -> {y}: base {base.mean():+.3f}  "
                  + "  ".join(f"{x} {effects[x]['E']:+.3f} (T {effects[x]['T']:.2f})" for x in CONTEXTS)
                  + f"  random max {max(rand_e):+.3f}", flush=True)  # fmt: skip
    RESULTS.mkdir(exist_ok=True)
    out(f"{name}.json").write_text(json.dumps(result, indent=1) + "\n")
    if not replies_out:
        return
    with out("replies.jsonl").open("w") as f:
        for r in replies_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    commands = {"extract": extract, "geometry": geometry, "steer-dev": steer_dev, "steer": steer,
                "extract-controls": extract_controls, "geometry-controls": geometry_controls,
                "steer-truth-capitals": steer}  # fmt: skip
    parser.add_argument("cmd", choices=list(commands))
    args = parser.parse_args()
    commands[args.cmd](args)


if __name__ == "__main__":
    main()
