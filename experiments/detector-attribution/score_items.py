"""Score every item sent to Pangram with the four open detectors, and occlude the splices word by word.

    python score_items.py   # -> results/scores/items.json {det: {item key: logit(score)}}, results/attr/splices.json

So each Pangram verdict sits beside the open detectors' verdicts on the identical text, and the splice windows can be
compared with each detector's word attributions.
"""

import budget
from attribute import _chunks, det
from common import DATA, DETECTORS, read_jsonl, save_json, words


def main() -> None:
    items = read_jsonl(DATA / "pangram_items.jsonl")
    spl = [it for it in items if it["group"] == "splice"]
    scores, occl = {}, {}
    for name in DETECTORS:
        d = det(name)

        def run(d=d):
            s = [round(v, 5) for c in _chunks([it["text"] for it in items], 300) for v in d.margins.remote(c)]
            o = [x for c in _chunks([it["text"] for it in spl], 10) for x in d.occlude.remote(c)]
            return s, o

        s, o = budget.paid_modal(f"score-items-{name}", 1.5, run)
        scores[name] = {it["key"]: v for it, v in zip(items, s, strict=True)}
        for it, x in zip(spl, o, strict=True):
            assert len(x["drop"]) == len(words(it["text"]))
            occl.setdefault(name, {})[it["key"]] = {"base": round(x["base"], 5), "w": x["drop"]}
        print(name, "done")
    save_json("scores/items.json", scores, indent=None)
    save_json("attr/splices.json", occl, indent=None)


if __name__ == "__main__":
    main()
