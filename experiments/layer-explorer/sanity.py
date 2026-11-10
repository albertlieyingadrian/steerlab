"""The sanity checks behind the layer explorer guide, on Qwen2.5-0.5B-Instruct (CPU, about a minute).

    .venv/bin/python experiments/layer-explorer/sanity.py

1. The split adds up: embed + every head + every neuron = the real target logit, with the final norm's 1/rms frozen
   (the leftover, on the six presets and two new prompts).
2. The logit lens at the last layer is the real output (largest difference over the whole vocabulary).
3. Unedited means untouched: an unedited trace is bit-identical to a plain forward pass, an edit of ×1 is too, and
   unedited greedy generation is token-identical to plain `generate` (repetition_penalty=1.0).
4. Dose against random: for every saved dose test in the explorer's data.json, whether the neuron's curve leaves the
   band of 20 random neurons from its layer, and by how much.
5. One gloss checked by hand: L21.N4090 promotes "法国 / French / France". Does it fire on France prompts and stay
   quiet on matched prompts about other countries? 10 + 10 prompts, peak activation over the prompt's tokens.

Writes results/sanity.json and prints every number the guide quotes.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import torch
from steerlab.inspect.core import Edit, Inspector
from steerlab.inspect.export import PRESETS

HERE = Path(__file__).resolve().parent
DATA = HERE / "report" / "explorer" / "data.json"
OUT = HERE / "results" / "sanity.json"
NEW_PROMPTS = ["The largest planet in the solar system is", "Water freezes at a temperature of zero degrees"]
GLOSS_NEURON = (21, 4090)
FRANCE = [
    "The capital of France is",
    "I spent the summer in France, eating",
    "The French president lives in",
    "Paris is the capital of",
    "My favourite French cheese is",
    "The Eiffel Tower is in",
    "She speaks fluent French and",
    "The Louvre museum is in Paris,",
    "Wine from Bordeaux, France, is",
    "The Tour de France is a",
]
OTHER = [
    "The capital of Germany is",
    "I spent the summer in Spain, eating",
    "The American president lives in",
    "Berlin is the capital of",
    "My favourite Italian cheese is",
    "The Colosseum is in",
    "She speaks fluent Japanese and",
    "The Prado museum is in Madrid,",
    "Wine from Napa, California, is",
    "The Giro d'Italia is a",
]


def main() -> None:
    inspector = Inspector.load()
    prompts = [p["text"] for p in PRESETS] + NEW_PROMPTS

    leftover = []
    lens_last = []
    identical = {"trace": 0, "scale_x1": 0, "generate": 0, "n": len(prompts)}
    for text in prompts:
        run = inspector.run(text)
        leftover.append({"prompt": text, "target": run["target"]["text"], "leftover": run["leftover"]})
        ids = inspector.tokenize(text)
        trace = inspector.trace(ids)
        with torch.no_grad():
            plain = inspector.model(torch.tensor([ids]), use_cache=False).logits[0]
            stream = trace.resid[-1][-1]
            lens = (stream * inspector._norm_scale(stream) * inspector.w_norm) @ inspector.W_U.T
        lens_last.append(
            {
                "prompt": text,
                "max_abs_diff": float((lens - plain[-1]).abs().max()),
                "same_top": int(lens.argmax()) == int(plain[-1].argmax()),
            }
        )
        identical["trace"] += int(torch.equal(trace.logits, plain))
        identical["scale_x1"] += int(torch.equal(inspector.trace(ids, [Edit(20, 0, "scale", 1.0)]).logits, plain))
        with torch.no_grad():
            reference = inspector.model.generate(
                torch.tensor([ids]),
                attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                max_new_tokens=12,
                do_sample=False,
                repetition_penalty=1.0,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=inspector.tokenizer.eos_token_id,
            )[0, len(ids) :].tolist()
        identical["generate"] += int(inspector.generate(ids, (), 12) == reference)

    dose = []
    for preset in json.loads(DATA.read_text())["presets"]:
        for neuron, d in preset["doses"].items():
            base = d["base_prob"]
            outside = [
                s
                for s, p, lo, hi in zip(d["scales"], d["curve"], d["band_lo"], d["band_hi"], strict=True)
                if p < lo or p > hi
            ]
            dose.append(
                {
                    "preset": preset["id"],
                    "neuron": neuron,
                    "outside_at": outside,
                    "own_swing": max(abs(p - base) for p in d["curve"]),
                    "random_swing": max(abs(p - base) for curve in d["random"] for p in curve),
                }
            )

    layer, n = GLOSS_NEURON

    def peak(text: str) -> float:
        acts = inspector.trace(inspector.tokenize(text)).acts[layer][:, n]
        return float(acts.max())

    gloss = {
        "neuron": f"L{layer}.N{n}",
        "promotes": inspector.neuron(FRANCE[0], layer, n, k=6)["promotes"],
        "france": [{"prompt": t, "max_act": peak(t)} for t in FRANCE],
        "other": [{"prompt": t, "max_act": peak(t)} for t in OTHER],
    }
    france_acts = [row["max_act"] for row in gloss["france"]]
    other_acts = [row["max_act"] for row in gloss["other"]]
    gloss["threshold"] = 5.0
    gloss["france_above"] = sum(a > gloss["threshold"] for a in france_acts)
    gloss["other_above"] = sum(a > gloss["threshold"] for a in other_acts)

    results = {
        "model": inspector.name,
        "date": date.today().isoformat(),
        "leftover": leftover,
        "lens_last": lens_last,
        "identical": identical,
        "dose": dose,
        "gloss": gloss,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")

    print(f"1. leftover: max |leftover| = {max(abs(r['leftover']) for r in leftover):.1e} over {len(leftover)} prompts")
    for r in leftover:
        print(f"   {r['leftover']:+.1e}  {r['target']!r:12}  {r['prompt']}")
    worst = max(r["max_abs_diff"] for r in lens_last)
    same = sum(r["same_top"] for r in lens_last)
    print(f"2. lens at the last layer vs real logits: max |diff| = {worst:.1e}, same top word {same}/{len(lens_last)}")
    print(
        f"3. bit-identical: trace {identical['trace']}/{identical['n']}, "
        f"×1 edit {identical['scale_x1']}/{identical['n']}, greedy generation {identical['generate']}/{identical['n']}"
    )
    leaves = sum(bool(r["outside_at"]) for r in dose)
    big = sum(r["own_swing"] > 2 * r["random_swing"] for r in dose)
    print(f"4. dose: {leaves}/{len(dose)} saved dose curves leave the random band; {big} swing more than twice as far")
    for r in dose:
        print(
            f"   {r['preset']:12} {r['neuron']:10} own ±{100 * r['own_swing']:.1f} pts, random ±"
            f"{100 * r['random_swing']:.1f} pts, outside at {r['outside_at']}"
        )
    print(
        f"5. gloss {gloss['neuron']} (promotes {gloss['promotes'][:4]}): peak > {gloss['threshold']} on "
        f"{gloss['france_above']}/10 France prompts, {gloss['other_above']}/10 others; mean "
        f"{sum(france_acts) / 10:.1f} vs {sum(other_acts) / 10:.1f}"
    )
    for a, b in zip(gloss["france"], gloss["other"], strict=True):
        print(f"   {a['max_act']:6.1f}  {a['prompt']:40}  {b['max_act']:6.1f}  {b['prompt']}")


if __name__ == "__main__":
    main()
