"""The precomputed alchemy table: every card's measured profile at every strength with real replies, and every measured
pair with its label, error, loop share and replies. Writes report/table_<model>.js (window.VA_TABLES[model]), loaded
by the note with plain <script> tags, so it works from file:// and on GitHub Pages with no backend.

    python build_table.py                  # Gemma-2-2B-it
    VA_MODEL=qwen python build_table.py    # Qwen3-8B
"""

from __future__ import annotations

import json
import math
from collections import defaultdict

import pipeline
import va

TRAITS = list(va.TRAITS)
DEMO_PI = [0, 4, 8, 12]
"""One held-out prompt per category (explain, advise, describe, message) for the example replies."""
CLIP = 700


def clip(text: str) -> str:
    return text if len(text) <= CLIP else text[:CLIP].rsplit(" ", 1)[0] + " …"


def r2(x):
    return None if x is None or math.isnan(x) else round(float(x), 2)


def main() -> None:
    a = json.loads((va.RESULTS / "analysis_heldout.json").read_text())
    meta = pipeline.meta()["units"]
    cpath, lpath = va.RESULTS / "circuits.json", va.RESULTS / "neuronpedia_labels.json"
    circ = json.loads(cpath.read_text()) if cpath.exists() else {}
    labels = json.loads(lpath.read_text()) if lpath.exists() else {}
    ex = defaultdict(dict)
    for r in pipeline.load_scored("heldout"):
        if r["pi"] in DEMO_PI:
            ex[(r["kind"], tuple(r["uids"]), tuple(r["s"]))][r["pi"]] = {
                "text": clip(r["text"]),
                "loops": r["loops"],
                "repeat": r["repeat"],
            }

    def examples(kinds, uids, s):
        for k in kinds:
            if (k, tuple(uids), tuple(s)) in ex:
                return ex[(k, tuple(uids), tuple(s))]
        return {}

    cards = []
    for key, u in a["units"].items():
        m = meta.get(u["uid"], {})
        info: dict = {}
        if u["type"] == "neuron" and "neuron" in m:
            info = {"what": f"MLP neuron {m['neuron'][1]} in layer {m['neuron'][0]}"}
        elif u["type"] == "sae" and va.TAG == "qwen":
            L, f = m["feature"]
            info = {"what": f"Qwen-Scope feature {f} (layer {L}, 64k features, 50 active per word)"}
        elif u["type"] == "sae":
            L, f = m["feature"]
            info = {
                "what": f"Gemma Scope feature {f} (layer {L}, 16k)",
                "link": f"https://www.neuronpedia.org/gemma-2-2b/{L}-gemmascope-res-16k/{f}",
                "np_label": labels.get(f"{L}/{f}"),
            }
        elif u["type"] == "community":
            lay = [x[0] for x in m["members"]]
            info = {"what": f"{m['size']} MLP neurons across layers {min(lay)}-{max(lay)}", "size": m["size"]}
        elif u["type"] == "direction":
            info = {"what": f"difference of mean activations, layer {m.get('L', '')}"}
        elif u["type"] == "circuit":
            feats = circ.get("traits", {}).get(u["trait"], [])
            info = {
                "what": (
                    f"{len(feats)} transcoder features across layers {min(x[0] for x in feats)}-"
                    f"{max(x[0] for x in feats)}"
                )
                if feats
                else "circuit"
            }
        elif u["type"] == "prompt":
            info = {"what": "the style instruction, added to the question"}
        if m.get("top_tokens"):
            info["top_tokens"] = m["top_tokens"][:8]
        by_s = {}
        for s, p in u["by_s"].items():
            if u["type"] == "prompt":
                e = examples(["prompt"], [u["uid"]], [])
            else:
                e = examples(["single", "single_full"], [u["uid"]], [float(s)])
            by_s[s] = {
                "d": [r2(p["d"][t]) for t in TRAITS],
                "coh": r2(p["coherence"]),
                "hon": r2(p["honesty"]),
                "loop": r2(p["loop_share"]),
                "ex": e,
            }
        cards.append(
            {
                "key": key,
                "type": u["type"],
                "trait": u["trait"],
                "s_op": u["s_op"],
                "by_s": by_s,
                "control": {s: r2(c["d"]) for s, c in u.get("control_by_s", {}).items()},
                "breadth": u.get("breadth"),
                "target_op": r2(u.get("target_op")),
                "top_by_s": u.get("top_trait_by_s"),
                **info,
            }
        )
    pairs = []
    for p in a["pairs"]:
        if p["kind"] != "pair":
            continue
        pairs.append(
            {
                "type": p["type"],
                "traits": p["traits"],
                "label": p["label"],
                "label_loose": p["label_loose"],
                "E": r2(p["E"]),
                "eligible": p["eligible"],
                "d_a": [r2(x) for x in p["d_a"]],
                "d_b": [r2(x) for x in p["d_b"]],
                "d_ab": [r2(x) for x in p["d_ab"]],
                "coh": r2(p["coherence"]),
                "hon": r2(p["honesty"]),
                "loop": r2(p["loop_share"]),
                "emergent": p.get("emergent_trait"),
                "cos": r2(p.get("cos")),
                "ex": examples(["pair"], p["uids"], p["s"]),
            }
        )
    types = {k: {kk: (r2(vv) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in a["types"].items()}
    table = {
        "traits": TRAITS,
        "trait_labels": [va.label(t) for t in TRAITS],
        "strengths": pipeline.FINAL_GRID,
        "prompts": {pi: va.HELDOUT_PROMPTS[pi][1] for pi in DEMO_PI},
        "unsteered": {"ex": examples(["unsteered"], [], []), "raw": {k: r2(v) for k, v in a["unsteered_raw"].items()}},
        "types": types,
        "cards": cards,
        "pairs": pairs,
    }
    table["model"] = va.MODEL
    out = va.HERE / "report" / f"table_{va.TAG}.js"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        f"(window.VA_TABLES = window.VA_TABLES || {{}})[{json.dumps(va.TAG)}] = "
        + json.dumps(table, separators=(",", ":"))
        + ";\n"
    )
    print(f"wrote {out.name}: {len(cards)} cards, {len(pairs)} pairs, {out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
