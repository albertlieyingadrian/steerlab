"""Draw the note's figures from results/, printing every number plotted.

    .venv/bin/python experiments/agent-traces/make_figures.py

curve.png      share of the 307 held-out agent tasks solved against the number of teacher traces trained on, ours and
               the guide's released runs, with 95% bootstrap intervals; the teacher as a line. The 120- and 241-trace
               runs stopped early (268 and 267 tasks finished) and are plotted on the tasks they finished.
forgetting.png IFEval prompt-level strict accuracy (541 prompts), MBPP pass@1 (257) and the share of MBPP replies that
               open a tool call with no tools offered, against the number of traces, with 95% bootstrap intervals.
diff.png       relative change of the residual stream, ||h_sft - h_base|| / ||h_base||, by layer and token role on 60
               held-out agent conversations, for our SFT-500 and the guide's; and at the decision token, with the
               difference between the two SFT runs as the floor.
recovery.png   the next action after a successful call, a harmless warning (decoy) or an injected error, base vs
               SFT-500 on 60 test contexts; and SFT-500's targeted fixes with the error direction or a random direction
               projected out.
bugfix.png     the 100 bug-fix tasks in the same agent, base vs SFT-500: tasks solved, runs that call the edit tool, runs
               that hit the 10-minute limit, with 95% bootstrap intervals; and the training traces' tool calls.
steer.png      SFT-500 after a successful call, with the error direction (or a random direction of the same size)
               added after one layer at 1, 4 and 16 times the error-minus-success distance: the share of next turns
               that answer now (n = 40), and at 16 times, the share of replies that talk about an error.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "report" / "figures"
EMBER, GREY, LIGHT, INK, BLUE = "#C9561C", "#6F6A63", "#C8C2B8", "#2B2723", "#3B6EA8"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "savefig.dpi": 150})  # fmt: skip


def boot(x: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    m = x[np.random.default_rng(seed).integers(0, len(x), (10_000, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def xpos(n: int, ours: bool) -> float:
    return np.log10(max(n, 40)) + (0.0 if ours else 0.03)


TICKS = ([40, 120, 241, 473, 963, 2238], ["0", "120", "241", "473", "963", "2,238"])
CURVE = [  # (traces after the 16k-token filter, run, ours?)
    (0, "ours-base", True), (0, "released-base", False),
    (120, "ours-sft125s42", True), (241, "ours-sft250s42", True),
    (473, "ours-sft500s42", True), (473, "released-sft-500", False),
    (963, "released-sft-1k", False), (2238, "released-sft-full", False),
]  # fmt: skip


def style(ours: bool) -> dict:
    return {"fmt": "o" if ours else "s", "color": EMBER if ours else GREY, "mfc": EMBER if ours else "white",
            "capsize": 3, "ms": 6}  # fmt: skip


def curve() -> None:
    runs = json.loads((RES / "summary.json").read_text())["runs"]
    cut = json.loads((RES / "curve_finished.json").read_text())  # runs stopped early: rate on the tasks they finished
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    print("curve.png: traces trained on, run, solved/tasks, rate [95% CI]")
    for n, run, ours in CURVE:
        if "both" not in runs.get(run, {}):
            continue
        r = dict(runs[run]["both"])
        if run in cut:
            r = {
                "rate": cut[run]["rate"],
                "ci": cut[run]["ci"],
                "solved": cut[run]["solved"],
                "n": cut[run]["finished"],
            }
        ax.errorbar(xpos(n, ours), 100 * r["rate"], yerr=[[100 * (r["rate"] - r["ci"][0])], [100 * (r["ci"][1] - r["rate"])]],
                     **style(ours))  # fmt: skip
        print(
            f"  {n:5d} {run:20s} {r['solved']:3d}/{r['n']} {100 * r['rate']:.1f} [{100 * r['ci'][0]:.1f}, {100 * r['ci'][1]:.1f}]"
        )
    t = runs["released-teacher"]["both"]
    ax.axhline(100 * t["rate"], color=LIGHT, lw=1, ls="--")
    ax.text(
        np.log10(42), 100 * t["rate"] + 2, f"teacher, Qwen3.5-397B-A17B: {100 * t['rate']:.0f}%", color=GREY, fontsize=8
    )
    print(f"  teacher {t['solved']}/{t['n']} {100 * t['rate']:.1f}")
    ax.set_xticks(np.log10(TICKS[0]), TICKS[1])
    ax.set_xlabel("teacher traces trained on (log scale; 0 = base model)")
    ax.set_ylabel("held-out tasks solved (%)")
    ax.set_ylim(0, 85)
    ax.errorbar([], [], **style(True), label="our runs")
    ax.errorbar([], [], **style(False), label="guide's released runs")
    ax.legend(frameon=False, loc="center left")
    fig.tight_layout()
    fig.savefig(FIG / "curve.png")


GENERAL = [(0, "base", True), (120, "sft125s42", True), (241, "sft250s42", True), (473, "sft500s42", True),
           (473, "released-sft500", False), (2238, "released-sftfull", False)]  # fmt: skip


def forgetting() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3))
    print("forgetting.png: traces, model, IFEval strict, MBPP pass@1, MBPP replies opening <tool_call>")
    for n, tag, ours in GENERAL:
        ife = [json.loads(x) for x in (RES / "general" / f"{tag}.ifeval.jsonl").read_text().splitlines()]
        mb = [json.loads(x) for x in (RES / "general" / f"{tag}.mbpp.jsonl").read_text().splitlines()]
        arrays = (np.array([r["strict"] for r in ife], float), np.array([r["pass"] for r in mb], float),
                  np.array(["<tool_call>" in r["text"] for r in mb], float))  # fmt: skip
        vals = []
        for ax, arr in zip(axes, arrays):
            m, lo, hi = boot(arr)
            ax.errorbar(xpos(n, ours), 100 * m, yerr=[[100 * (m - lo)], [100 * (hi - m)]], **style(ours))
            vals.append(f"{int(arr.sum())}/{len(arr)} ({100 * m:.1f} [{100 * lo:.1f}, {100 * hi:.1f}])")
        print(f"  {n:5d} {tag:17s} " + "  ".join(vals))
    titles = (
        "IFEval, prompt-level strict (%)",
        "MBPP pass@1 (%)",
        "MBPP replies that open a\ntool call, no tools offered (%)",
    )
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xticks(np.log10([40, 120, 241, 473, 2238]), ["0", "120", "241", "473", "2,238"], fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_xlabel("traces trained on", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "forgetting.png")


def diff() -> None:
    d = json.loads((RES / "interp" / "diff.json").read_text())
    r1, r2 = d["in_dist_full"]["sft - base"], d["in_dist_full"]["sft2 - base"]
    seeds = d["in_dist_full"]["sft2 - sft"]["rel_decision"]
    ch = d["chat"]["sft - base"]["rel_decision"]
    L = np.arange(1, len(seeds) + 1)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.2, 3.5))
    colours = {"system": EMBER, "assistant": "#E29A6B", "user": GREY, "tool": LIGHT}
    labels = {"system": "agent system prompt + tool schemas", "assistant": "the agent's own turns",
              "user": "task text", "tool": "tool outputs"}  # fmt: skip
    for role in ("system", "assistant", "user", "tool"):
        ax.plot(L, r1["rel_by_role"][role], color=colours[role], lw=2, label=labels[role])
        ax.plot(L, r2["rel_by_role"][role], color=colours[role], lw=1, ls=":")
    ax.set_xlabel("layer")
    ax.set_ylabel("||h_SFT − h_base|| / ||h_base||")
    ax.set_title(
        "every token of 60 held-out agent conversations\n(solid: our SFT-500; dotted: the guide's)", fontsize=9
    )
    ax.legend(frameon=False, fontsize=8)
    bx.plot(L, r1["rel_decision"], color=EMBER, lw=2, label="SFT vs base, agent prompt")
    bx.plot(L, ch, color="#E29A6B", lw=2, label="SFT vs base, plain chat prompt")
    bx.plot(L, seeds, color=BLUE, lw=2, label="our SFT-500 vs the guide's")
    bx.set_xlabel("layer")
    bx.set_ylim(0, 1)
    bx.set_title("at the last prompt token (where the reply starts)", fontsize=9)
    bx.legend(frameon=False, fontsize=8)
    print("diff.png: layer, relative change by role (ours), decision token: agent / chat / between runs")
    for i in range(0, len(L), 4):
        print(f"  L{i + 1:2d} system {r1['rel_by_role']['system'][i]:.2f} assistant {r1['rel_by_role']['assistant'][i]:.2f} "
              f"user {r1['rel_by_role']['user'][i]:.2f} tool {r1['rel_by_role']['tool'][i]:.2f} | "
              f"{r1['rel_decision'][i]:.2f} / {ch[i]:.2f} / {seeds[i]:.2f}")  # fmt: skip
    fig.tight_layout()
    fig.savefig(FIG / "diff.png")


def recovery() -> None:
    p = json.loads((RES / "interp" / "probe.json").read_text())
    beh = p["behaviour"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    conds = (
        ("success", "after a success"),
        ("decoy", "after a harmless warning"),
        ("error", "after an injected error"),
    )
    metrics = (("answer_now", "answers now"), ("targeted_fix", "targeted fix"), ("near_repeat", "repeats the call"))
    w = 0.12
    print("recovery.png: model/condition, answers now, targeted fix, repeats (% of 60)")
    for i, (cond, _) in enumerate(conds):
        for j, (model, colour) in enumerate((("base", GREY), ("sft", EMBER))):
            v = beh[f"{model}/{cond}"]
            for k, (m, _) in enumerate(metrics):
                x = i + (k - 1) * 0.3 + (j - 0.5) * w
                ax.bar(x, 100 * v[m][0], w, color=colour)
                ax.plot([x, x], [100 * v[m][1], 100 * v[m][2]], color=INK, lw=0.8)
            print(
                f"  {model}/{cond}: "
                + "  ".join(f"{m} {100 * v[m][0]:.0f} [{100 * v[m][1]:.0f}, {100 * v[m][2]:.0f}]" for m, _ in metrics)
            )
    ax.set_xticks([i + (k - 1) * 0.3 for i in range(3) for k in range(3)],
                  [lab for _ in conds for _, lab in metrics], fontsize=7, rotation=30, ha="right")  # fmt: skip
    for i, (_, title) in enumerate(conds):
        ax.text(i, 103, title, ha="center", fontsize=8, color=INK)
    ax.set_ylim(0, 110)
    ax.set_ylabel("next turns (%, n = 60)")
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=GREY, label="base"), Patch(color=EMBER, label="SFT-500")], frameon=False,
              fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.93))  # fmt: skip
    ab = p.get("ablation", {})
    order = ["probe"] + sorted(k for k in ab if k != "probe")
    vals = [("none", beh["sft/error"]["targeted_fix"], INK)] + [
        (k, ab[k]["targeted_fix"], EMBER if k == "probe" else LIGHT) for k in order if k in ab
    ]
    for i, (name, v, colour) in enumerate(vals):
        bx.bar(i, 100 * v[0], 0.6, color=colour)
        bx.plot([i, i], [100 * v[1], 100 * v[2]], color=INK, lw=0.8)
        print(f"  ablation {name}: targeted fix {100 * v[0]:.0f} [{100 * v[1]:.0f}, {100 * v[2]:.0f}]")
    bx.set_xticks(
        range(len(vals)), ["none", "error\ndirection"] + [f"random\n{i}" for i in range(1, len(vals) - 1)], fontsize=7
    )
    bx.set_ylim(0, 100)
    bx.set_ylabel("targeted fixes (%, n = 60)")
    bx.set_title("SFT-500 after an injected error,\nwith one direction projected out of every layer", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "recovery.png")


def steer() -> None:
    p = json.loads((RES / "interp" / "probe.json").read_text())
    st = p["steer"]
    dirs = sorted({k.split("@")[0] for k in st})
    scales = sorted({float(k.split("@")[1]) for k in st})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.6, 3.2), gridspec_kw={"width_ratios": [1.4, 1]})
    print("steer.png: direction, multiple of the gap, answers now % [CI]; error words at the largest push")
    for d in dirs:
        ys = [st[f"{d}@{sc:.3f}"]["answer_now"] for sc in scales]
        colour, lw = (EMBER, 2.2) if d == "probe" else (LIGHT, 1.4)
        xs = np.arange(len(scales)) + (0.04 if d != "probe" else 0)
        ax.errorbar(xs, [100 * y[0] for y in ys], yerr=[[100 * (y[0] - y[1]) for y in ys], [100 * (y[2] - y[0]) for y in ys]],
                    color=colour, lw=lw, marker="o", capsize=3, label="error direction" if d == "probe" else None)  # fmt: skip
        print(
            f"  {d}: "
            + "  ".join(f"{m:g}x {100 * y[0]:.0f} [{100 * y[1]:.0f}, {100 * y[2]:.0f}]" for m, y in zip((1, 4, 16), ys))
        )
    ax.plot([], [], color=LIGHT, marker="o", label="random directions (3)")
    ax.set_xticks(range(len(scales)), ["1x", "4x", "16x"])
    ax.set_xlabel("push, in multiples of the error-minus-success distance")
    ax.set_ylabel("next turns that answer now (%, n = 40)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("after a successful call", fontsize=9)
    ew = p["steer_error_words_at_max"]
    names = ["probe"] + [d for d in dirs if d != "probe"]
    for i, d in enumerate(names):
        k, n = ew[d]
        bx.bar(i, 100 * k / n, 0.6, color=EMBER if d == "probe" else LIGHT)
        print(f"  error words at 16x, {d}: {k}/{n}")
    bx.set_xticks(range(len(names)), ["error\ndirection"] + [f"random\n{i}" for i in range(1, len(names))], fontsize=8)
    bx.set_ylim(0, 100)
    bx.set_ylabel("replies naming an error (%)")
    bx.set_title("at 16x the replies stop being agent turns;\nthose that name an error", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "steer.png")


def bugfix() -> None:
    rows = {m: [json.loads(x) for x in (RES / "per_task" / f"ours-{m}.bugfix.jsonl").read_text().splitlines()]
            for m in ("base", "sft500s42")}  # fmt: skip
    traces = json.loads((RES / "trace_tools.json").read_text())
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.8, 3.2), gridspec_kw={"width_ratios": [1.6, 1]})
    metrics = (("solved", "tasks solved"), ("edit", "runs that call edit"), ("timeout", "runs that time out"))
    print("bugfix.png: model, solved, uses edit, timeouts (of 100) [CI]")
    for j, (m, colour, label) in enumerate((("base", GREY, "base"), ("sft500s42", EMBER, "SFT-500"))):
        for k, (key, _) in enumerate(metrics):
            x = np.array([(r["solved"] if key == "solved" else ("edit" in r["tool_names"]) if key == "edit" else r["timeout"])
                          for r in rows[m]], float)  # fmt: skip
            mean, lo, hi = boot(x)
            pos = k + (j - 0.5) * 0.36
            ax.bar(pos, 100 * mean, 0.34, color=colour, label=label if k == 0 else None)
            ax.plot([pos, pos], [100 * lo, 100 * hi], color=INK, lw=0.8)
            print(f"  {m} {key}: {int(x.sum())}/100 [{100 * lo:.0f}, {100 * hi:.0f}]")
    ax.set_xticks(range(3), [lab for _, lab in metrics], fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of 100 bug-fix runs")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Python bug fixes, same agent (OpenCode)", fontsize=9)
    names = list(traces["calls"])
    bx.barh(range(len(names)), [traces["calls"][n] for n in names], color=GREY)
    bx.set_yticks(range(len(names)), names, fontsize=8)
    bx.invert_yaxis()
    bx.set_xlabel("tool calls in the 497 training traces", fontsize=8)
    bx.set_title("what the traces show", fontsize=9)
    print("  training traces' tool calls:", traces["calls"])
    fig.tight_layout()
    fig.savefig(FIG / "bugfix.png")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (curve, forgetting, bugfix, diff, recovery, steer):
        f()
