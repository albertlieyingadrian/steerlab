"""Every number in the FinanceMath post-training note, recomputed from results/runs/: accuracy with bootstrap
intervals, and paired bootstrap differences (same questions) against base Qwen3-8B, gpt-5.4-mini and gpt-5.4.

    fm_summary.py --systems sft-e2 grpo-sft-e2-s10      # run tags without the split prefix
Writes results/summary.json and prints a markdown table per split.
The `training` block (best-of-4 counts, GRPO histories, SFT losses) is rebuilt from the trace and checkpoint files
when they are present, and kept as saved otherwise.
"""

from __future__ import annotations

import argparse
import json

import fm  # noqa: I001
import fin

RIVALS = ["qwen", "gpt-5.4-mini", "gpt-5.4"]


def load(split: str, tag: str) -> dict[str, bool] | None:
    path = fm.RESULTS / "runs" / f"{split}__{tag}.jsonl"
    if not path.exists():
        return None
    return {r["id"]: r["correct"] for r in map(json.loads, path.read_text().splitlines())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--systems", nargs="+", required=True)
    a = p.parse_args()
    out = {}
    for split in ("validation", "test"):
        runs = {t: load(split, t) for t in dict.fromkeys(RIVALS + a.systems)}
        runs = {t: r for t, r in runs.items() if r}
        rows = []
        for tag, res in runs.items():
            ids = sorted(res)
            row = {"system": tag, "n": len(ids), "acc": fin.bootstrap([res[i] for i in ids])}
            for ref in RIVALS:
                if ref != tag and ref in runs:
                    both = [i for i in ids if i in runs[ref]]
                    row[f"vs {ref}"] = fin.paired_bootstrap([runs[ref][i] for i in both], [res[i] for i in both])
            rows.append(row)
        out[split] = rows
        print(f"\n### {split}\n")
        print("| system | n | accuracy [95% CI] | " + " | ".join(f"vs {r}" for r in RIVALS) + " |")
        print("|---|---|---|" + "---|" * len(RIVALS))
        for row in rows:
            diffs = [row.get(f"vs {r}") for r in RIVALS]
            cells = [f"{d[0] * 100:+.1f} [{d[1] * 100:+.1f}, {d[2] * 100:+.1f}]" if d else "—" for d in diffs]
            print(f"| {row['system']} | {row['n']} | {fin.fmt_ci(row['acc'])} | " + " | ".join(cells) + " |")
    out["training"] = training()
    (fm.RESULTS / "summary.json").write_text(json.dumps(out, indent=1))


def training() -> dict:
    """What the figures need from the trace and checkpoint files, without checkpoint paths. Those files are not
    published; without them the counts already saved in results/summary.json are kept."""
    import collections

    saved = fm.RESULTS / "summary.json"
    t = json.loads(saved.read_text()).get("training", {}) if saved.exists() else {}
    data = fm.HERE / "data"
    for name, path in (("qwen_best_of_4", data / "qwen_bon.jsonl"), ("teacher_extra_3", data / "teacher_bon.jsonl")):
        if path.exists():
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            right = collections.Counter(sum(s["correct"] for s in r["samples"]) for r in rows)
            t[name] = {"problems": len(rows), "right_of_n": {str(k): right[k] for k in sorted(right)},
                       "identical_sample_sets": sum(len({s["text"] for s in r["samples"]}) == 1 for r in rows)}  # fmt: skip
    ckpt = fm.RESULTS / "posttrain" / "checkpoints.json"
    if ckpt.exists():
        log = json.loads(ckpt.read_text())
        t["grpo"] = {k: log[k]["history"] for k in ("grpo-base-s20", "grpo-rft-e1-s20") if k in log}
        t["sft_losses"] = {k: {"first": log[k]["losses"][0], "last": log[k]["losses"][-1], "examples": log[k]["examples"],
                               "steps": log[k]["steps"]} for k in ("sft-e2", "rft-e2") if k in log}  # fmt: skip
    return t


if __name__ == "__main__":
    main()
