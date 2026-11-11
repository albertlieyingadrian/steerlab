"""Step 5: does the push just break the model? Free-text replies under steering, at the headline layer.

Eight everyday prompts, neutral system prompt, greedy decoding (repetition_penalty=1.0), 64 new tokens, the vector
added at every position (prompt and generated). Sources: awareness pushed up (+) and down (-), the unrelated
direction and one random direction, all at the same length, at every strength. Scores per reply:
    ppl        perplexity of the reply under the UNSTEERED model (high = garbled)
    distinct2  share of distinct word pairs in the reply (low = looping)
A reply is "fluent" when ppl < 3x the unsteered replies' median and distinct2 >= 0.6.
Writes results/<model>/fluency.csv and results/<model>/fluency_samples.json (one reply per cell).
"""

import csv
import itertools
import json
import sys

import numpy as np

import common as C
from steer import ALPHAS, vectors


def generate(model, tok, texts, max_new=64):
    import torch

    outs = []
    for t in texts:
        x = tok(t, return_tensors="pt", add_special_tokens=False).to(C.DEVICE)
        with torch.inference_mode():
            g = model.generate(
                **x,
                max_new_tokens=max_new,
                do_sample=False,
                repetition_penalty=1.0,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        outs.append(tok.decode(g[0, x["input_ids"].shape[1] :], skip_special_tokens=True))
    return outs


def ppl(model, tok, prompt_text, reply):
    import torch

    p = tok(prompt_text, add_special_tokens=False)["input_ids"]
    r = tok(reply, add_special_tokens=False)["input_ids"]
    if not r:
        return float("inf")
    x = torch.tensor([p + r], device=C.DEVICE)
    with torch.inference_mode():
        lg = model(input_ids=x).logits[0, len(p) - 1 : -1].float()
    nll = torch.nn.functional.cross_entropy(lg, torch.tensor(r, device=C.DEVICE))
    return float(torch.exp(nll))


def distinct2(text):
    w = text.split()
    pairs = list(itertools.pairwise(w))
    return len(set(pairs)) / len(pairs) if pairs else 0.0


def main(name, layer=None):
    tok, model = C.load(name)
    if layer is None:
        layer = json.loads((C.RESULTS / name / "verdict.json").read_text())["headline_layer"]
    layer = int(layer)
    src, norm = vectors(name, layer)
    texts = [
        tok.apply_chat_template(
            [{"role": "system", "content": C.NEUTRAL_SYSTEM}, {"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
            **C.TEMPLATE_KWARGS.get(name, {}),
        )
        for q in C.FLUENCY_PROMPTS
    ]
    arms = [("none", 0.0, None)]
    for a in ALPHAS:
        arms += [
            ("awareness+", a, a * norm * src["awareness"]),
            ("awareness-", a, -a * norm * src["awareness"]),
            ("unrelated", a, a * norm * src["unrelated"]),
            ("rand0", a, a * norm * src["rand0"]),
        ]
    rows, samples = [], {}
    for arm, a, vec in arms:
        with C.Steer(model, layer, vec):
            replies = generate(model, tok, texts)
        for i, r in enumerate(replies):
            rows.append(
                {
                    "arm": arm,
                    "alpha": a,
                    "prompt": i,
                    "ppl": ppl(model, tok, texts[i], r),
                    "distinct2": distinct2(r),
                    "chars": len(r),
                }
            )
        samples[f"{arm}@{a}"] = replies[0]
        print(
            arm,
            a,
            round(float(np.median([x["ppl"] for x in rows if x["arm"] == arm and x["alpha"] == a])), 2),
            repr(replies[0][:90]),
            flush=True,
        )
    med0 = float(np.median([r["ppl"] for r in rows if r["arm"] == "none"]))
    for r in rows:
        r["fluent"] = bool(r["ppl"] < 3 * med0 and r["distinct2"] >= 0.6)
    with open(C.RESULTS / name / "fluency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    C.write_json(
        C.RESULTS / name / "fluency_samples.json",
        {"layer": layer, "prompt": C.FLUENCY_PROMPTS[0], "unsteered_median_ppl": med0, "replies": samples},
    )


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3] or []))
