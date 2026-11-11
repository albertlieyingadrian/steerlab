"""Solve rates and paired bootstrap intervals for every run in results/per_task/.

    .venv/bin/python experiments/agent-traces/analyze.py            # prints a table, writes results/summary.json

Denominators are the frozen splits (data/splits/): 144 eval tasks and the 163 test tasks (and the 100 bug-fix tasks); a task with no result
counts as unsolved. "both" pools the 307. Intervals: 95% percentile bootstrap over tasks (10,000 resamples, seed 0);
differences between two runs resample the same tasks for both (paired). Behaviour columns are over the tasks with a
result: answer written, finished (last turn without a tool call), timeout (the agent hit the task's time limit), median turns.
"""

import json
import statistics
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPLITS = HERE / "data" / "splits"
B = 10_000


def split_indices() -> dict[str, list[int]]:
    test = [int(x) for x in (SPLITS / "test_indices.txt").read_text().strip().split(",")]
    return {"eval": list(range(144)), "test": test, "bugfix": list(range(100))}


def load_runs() -> dict[str, dict[str, dict[int, dict]]]:
    runs: dict = {}
    for f in sorted((HERE / "results" / "per_task").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            runs.setdefault(r["run"], {}).setdefault(r["split"], {})[r["index"]] = r
    return runs


def vector(run: dict, split: str, idx: dict[str, list[int]]) -> np.ndarray | None:
    if split == "both":
        parts = [vector(run, s, idx) for s in ("eval", "test")]
        return None if any(p is None for p in parts) else np.concatenate(parts)
    rows = run.get(split)
    if not rows:
        return None
    return np.array([rows.get(i, {}).get("solved", 0) for i in idx[split]], dtype=float)


def boot(x: np.ndarray, y: np.ndarray | None = None, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    d = x if y is None else x - y
    means = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def behaviour(run: dict, split: str, idx: list[int]) -> dict:
    """Behaviour rates over the tasks with a result file (rows rebuilt from a run log carry only the reward)."""
    rows = [run[split][i] for i in idx if i in run.get(split, {})]
    full = [r for r in rows if not r.get("from_log")]
    if not full:
        return {}
    n = len(full)
    return {
        "with_result": len(rows),
        "with_result_file": n,
        "answer_written": sum(r["answer_written"] for r in full) / n,
        "finished": sum(r["finished"] for r in full) / n,
        "timeout": sum(r["timeout"] for r in full) / n,
        "median_turns": statistics.median(r["n_turns"] for r in full),
    }


def main():
    idx = split_indices()
    runs = load_runs()
    out: dict = {"runs": {}, "paired": {}}
    print(f"{'run':28s} {'split':5s} {'solved':>9s}  rate [95% CI]           answer  timeout  turns")
    for name, run in runs.items():
        out["runs"][name] = {}
        for s in ("eval", "test", "both", "bugfix"):
            v = vector(run, s, idx)
            if v is None:
                continue
            m, lo, hi = boot(v)
            b = behaviour(run, s, idx[s]) if s != "both" else {}
            out["runs"][name][s] = {"solved": int(v.sum()), "n": len(v), "rate": m, "ci": [lo, hi], **b}
            extra = f"{b['answer_written']:6.2f}  {b['timeout']:7.2f}  {b['median_turns']:5.1f}" if b else ""
            print(f"{name:28s} {s:5s} {int(v.sum()):4d}/{len(v):<4d} {m:6.3f} [{lo:.3f}, {hi:.3f}]  {extra}")
    names = sorted(runs)
    for i, a in enumerate(names):
        for b_ in names[i + 1 :]:
            for s in ("eval", "test", "both", "bugfix"):
                xa, xb = vector(runs[a], s, idx), vector(runs[b_], s, idx)
                if xa is None or xb is None:
                    continue
                m, lo, hi = boot(xa, xb)
                out["paired"][f"{a} - {b_} | {s}"] = {"diff": m, "ci": [lo, hi],
                                                     "a_only": int(((xa == 1) & (xb == 0)).sum()),
                                                     "b_only": int(((xa == 0) & (xb == 1)).sum())}  # fmt: skip
    (HERE / "results" / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print("\npaired differences (a - b) on the 307 tasks:")
    for k, v in out["paired"].items():
        if k.endswith("| both") or k.endswith("| bugfix"):
            lo, hi = v["ci"]
            print(
                f"  {k:52s} {v['diff']:+.3f} [{lo:+.3f}, {hi:+.3f}]  a-only {v['a_only']:3d}  b-only {v['b_only']:3d}"
            )


def completed() -> None:
    """Runs cut short (the 120- and 241-trace evaluations stopped when the compute account was disabled): solve rates
    on the tasks they finished, and paired differences against other runs on those same tasks."""
    idx = split_indices()
    runs = load_runs()
    keys = [("eval", i) for i in idx["eval"]] + [("test", i) for i in idx["test"]]
    out: dict = {}
    print("\nruns cut short: rate on finished tasks, and paired against others on the same tasks")
    for name in ("ours-sft125s42", "ours-sft250s42"):
        if name not in runs:
            continue
        done = [k for k in keys if k[1] in runs[name].get(k[0], {})]
        x = np.array([runs[name][s][i]["solved"] for s, i in done], float)
        m, lo, hi = boot(x)
        out[name] = {"finished": len(done), "of": len(keys), "solved": int(x.sum()), "rate": m, "ci": [lo, hi],
                     "lower_bound_rate": float(x.sum() / len(keys)), "paired": {}}  # fmt: skip
        print(
            f"  {name}: {int(x.sum())}/{len(done)} finished = {100 * m:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]; "
            f"missing counted unsolved {100 * x.sum() / len(keys):.1f}"
        )
        for other in ("ours-base", "ours-sft500s42", "released-sft-500", "ours-sft125s42", "ours-sft250s42"):
            if other == name or other not in runs:
                continue
            y = np.array([runs[other].get(s, {}).get(i, {}).get("solved", 0) for s, i in done], float)
            dm, dlo, dhi = boot(x, y)
            out[name]["paired"][other] = {"diff": dm, "ci": [dlo, dhi], "other_rate": float(y.mean())}
            print(
                f"     - {other:18s} {100 * dm:+.1f} [{100 * dlo:+.1f}, {100 * dhi:+.1f}] (other {100 * y.mean():.1f} on these)"
            )
    (HERE / "results" / "curve_finished.json").write_text(json.dumps(out, indent=1) + "\n")


def general() -> None:
    """Forgetting checks: each model against the base on the same prompts (paired), from results/general/."""
    gdir = HERE / "results" / "general"
    if not (gdir / "base.ifeval.jsonl").exists():
        return
    out: dict = {}
    print("\nforgetting checks, paired against the base (points):")
    for bench, key, ident in (("ifeval", "strict", "key"), ("mbpp", "pass", "id")):
        base = {r[ident]: r for r in map(json.loads, (gdir / f"base.{bench}.jsonl").read_text().splitlines())}
        for f in sorted(gdir.glob(f"*.{bench}.jsonl")):
            tag = f.name.split(".")[0]
            rows = {r[ident]: r for r in map(json.loads, f.read_text().splitlines())}
            x = np.array([rows[i][key] for i in base], float)
            d = x - np.array([base[i][key] for i in base], float)
            m, lo, hi = boot(x)
            dm, dlo, dhi = boot(d)
            words = statistics.median(len(rows[i]["text"].split()) for i in base)
            out[f"{tag}/{bench}"] = {"n": len(x), "right": int(x.sum()), "rate": m, "ci": [lo, hi], "diff": dm,
                                     "diff_ci": [dlo, dhi], "broke": int((d < 0).sum()), "fixed": int((d > 0).sum()),
                                     "tool_call_replies": sum("<tool_call>" in rows[i]["text"] for i in base),
                                     "median_words": words}  # fmt: skip
            print(
                f"  {tag:17s} {bench:6s} {int(x.sum()):3d}/{len(x)}  {100 * dm:+5.1f} [{100 * dlo:+.1f}, {100 * dhi:+.1f}]"
                f"  broke {int((d < 0).sum())} fixed {int((d > 0).sum())}  tool calls {out[f'{tag}/{bench}']['tool_call_replies']}"
            )
    (HERE / "results" / "general_summary.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
    completed()
    general()
