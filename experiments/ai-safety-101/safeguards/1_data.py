"""
Safeguards, script 1: harmful vs harmless prompts, split three ways, plus the red team's rewrites.

Rebuilds the data side of "Constitutional Classifiers++" (Cunningham, Wei et al. 2026, arXiv 2601.04603) at laptop
scale. Prompts only: nothing here asks any model to answer them, and the prompt text stays in the local cache
~/.cache/steerlab/ai-safety-101/05/prompts.json (git gets ids, labels and scores).

Splits (fixed seed, by source item, so a rewrite of a test prompt never leaks into training):
- train: 260 AdvBench + 400 Dolly. Only the linear probe learns from these.
- calib: 500 Dolly. Every classifier's threshold is set here, at a 1% false-positive rate (FPR): the defender
  calibrates on normal traffic. No harmful prompts, so thresholds never see the attacks.
- test: 260 AdvBench + 400 Dolly, each in 6 versions (clean plus 5 red-team rewrites), and XSTest's 250 safe and 200
  unsafe prompts, clean only (hard cases from a different source).

Red-team rewrites (deterministic, in guard.TRANSFORMS): synonym swaps, a role-play wrapper, base64, leetspeak, and a
translate-then-answer wrapper. The same rewrites applied to harmless prompts are the control: do they trip the
classifier on their own?

Run: python experiments/ai-safety-101/safeguards/1_data.py   (seconds; downloads ~13 MB once)
"""

import guard as g
import numpy as np

import tools as t

RESULT = g.RESULTS / "1_data.json"


def main():
    adv, dolly, xs = g.load_raw()
    rng = np.random.default_rng(t.SEED)
    adv_order = rng.permutation(len(adv))
    dolly_pick = rng.choice(len(dolly), size=1300, replace=False)

    items = []

    def add(source, index, split, label, prompt, transforms=("clean",)):
        for name in transforms:
            items.append(
                {
                    "id": f"{source}-{index:05d}:{name}",
                    "base_id": f"{source}-{index:05d}",
                    "source": source,
                    "split": split,
                    "transform": name,
                    "label": label,
                    "prompt": g.TRANSFORMS[name](prompt),
                }
            )

    for k, i in enumerate(adv_order):
        split = "train" if k < 260 else "test"
        add("advbench", int(i), split, 1, adv[i], g.TRANSFORMS if split == "test" else ("clean",))
    for k, j in enumerate(dolly_pick):
        row, text = dolly[j]
        split = "train" if k < 400 else "calib" if k < 900 else "test"
        add("dolly", row, split, 0, text, g.TRANSFORMS if split == "test" else ("clean",))
    for row in xs:
        add("xstest", int(row["id"]), "test", int(row["label"] == "unsafe"), row["prompt"])
    t.write_json(g.PROMPTS, items)

    def count(**conds):
        return sum(all(x[k] == v for k, v in conds.items()) for x in items)

    print(f"Pool: {len(adv)} AdvBench, {len(dolly)} Dolly instructions with no context, {len(xs)} XSTest.\n")
    print(f"{'split':>6s} {'source':>9s} {'label':>9s} {'base prompts':>13s} {'with rewrites':>14s}")
    counts = {}
    for split, source, label in (
        ("train", "advbench", 1),
        ("train", "dolly", 0),
        ("calib", "dolly", 0),
        ("test", "advbench", 1),
        ("test", "dolly", 0),
        ("test", "xstest", 0),
        ("test", "xstest", 1),
    ):
        base = count(split=split, source=source, label=label, transform="clean")
        total = count(split=split, source=source, label=label)
        counts[f"{split}/{source}/{label}"] = {"base": base, "total": total}
        print(f"{split:>6s} {source:>9s} {'harmful' if label else 'harmless':>9s} {base:>13d} {total:>14d}")

    test_adv = [x for x in items if x["source"] == "advbench" and x["split"] == "test"]
    clean = {x["base_id"]: x["prompt"] for x in test_adv if x["transform"] == "clean"}
    syn = [x for x in test_adv if x["transform"] == "synonyms"]
    changed = float(np.mean([x["prompt"] != clean[x["base_id"]] for x in syn]))
    print(f"\nSynonym swaps changed {changed:.0%} of the harmful test prompts (the rest had no listed word).")

    example = next(x["prompt"] for x in items if x["split"] == "test" and x["source"] == "dolly")
    print("\nThe five rewrites on one harmless test prompt (the harmful ones get exactly the same treatment):")
    for name, fn in g.TRANSFORMS.items():
        print(f"  [{name}] {fn(example)!r}")

    train_ids = {x["base_id"] for x in items if x["split"] == "train"}
    leaks = train_ids & {x["base_id"] for x in items if x["split"] != "train"}
    t.write_json(
        RESULT,
        {
            "counts": counts,
            "synonym_changed_share": changed,
            "n_items": len(items),
            "sources": {k: {"url": u, "sha256": s} for k, (u, s) in g.SOURCES.items()},
        },
    )
    print("\nSelf-checks")
    t.check("no source prompt appears in more than one split", not leaks)
    t.check(
        "every test AdvBench and Dolly prompt has all 6 versions",
        count(split="test", source="advbench") == 6 * 260 and count(split="test", source="dolly") == 6 * 400,
    )
    t.check("item ids are unique", len({x["id"] for x in items}) == len(items))
    t.check(f"prompt text written only to the local {t.rel(g.PROMPTS)}", g.PROMPTS.exists())


if __name__ == "__main__":
    main()
