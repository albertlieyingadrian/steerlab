"""Draw the Episode 1 note's figures from results/, printing every number plotted.

    .venv/bin/python experiments/graft-monitor/make_figures.py

Inputs: results/ep1_summary.json (analyze_ep1.py), results/monitor_bench/summary.json (monitors.bench summary), and
for the training curves the saved eval transcripts in results/runs/ep1_*/evals/step_*.jsonl, re-graded with the
same parser (env.grade, v2) as analyze_ep1.py. Only finished runs (done.json present) are drawn, as in the summary.

arms.png      deployed hack rate against skill, one point per arm (mean of seeds, 95% bootstrap intervals), base marked.
modes.png     GRAFT and GRAFT without routing: hack rate with both adapters, RETAIN only and FORGET only.
recall.png    deployed hack rate against the share of hacks flagged (recall sweep), per seed and mean, with the base
              model and random flags at rate 0.4 as reference lines.
monitors.png  recall and false-positive rate of every monitor on the 400 labelled held-out episodes, 95% intervals.
curves.png    deployed hack rate over training steps (every 25), mean over finished seeds per arm.
hero.jpg, og.jpg  the cover (16:9) and its 1200x630 share image: every finished run's deployed hack rate over training,
              one line per seed, no axes; plus the research page's card, ../shared/thumbs/graft-monitor.jpg.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import env  # noqa: E402
from analyze_ep1 import GRAFT_FAMILY, regrade  # noqa: E402

RESULTS = HERE / "results"
RUNS = RESULTS / "runs"
FIGURES = HERE / "report" / "figures"
EMBER, GREY, LIGHT, INK = "#C9561C", "#6F6A63", "#C8C2B8", "#2B2723"
PAPER = "#FAF7F2"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "savefig.dpi": 150})  # fmt: skip

ARMS = [  # (group, label, colour, marker, line style for curves)
    ("none", "no intervention", GREY, "o", "-"),
    ("penalty", "reward penalty", GREY, "s", "--"),
    ("filter", "filtering", GREY, "^", ":"),
    ("graft_noroute", "GRAFT without routing", GREY, "D", "-."),
    ("graft", "GRAFT", EMBER, "o", "-"),
]
MONITORS = [  # (backend, label)
    ("oss_safeguard", "gpt-oss-safeguard-20b, medium"),
    ("oss_safeguard@low", "gpt-oss-safeguard-20b, low (routing)"),
    ("qwen_judge", "Qwen3-8B judge"),
    ("control_arena", "control-arena, policy prompt"),
    ("control_arena@library_default", "control-arena, library default"),
    ("petri", "Petri judge"),
    ("rule", "phrase rule"),
]


def r3(xs) -> list[float]:
    return [round(x, 3) for x in xs]


def yerr(m: dict, key: str = "hack_rate", ci: str = "ci_hack") -> list[list[float]]:
    return [[m[key] - m[ci][0]], [m[ci][1] - m[key]]]


def finished_runs(group: str) -> list[Path]:
    """Finished runs of one arm (ep1_<arm>_s<seed>); the recall-sweep and random-flag runs are named apart."""
    return [d for d in sorted(RUNS.glob(f"ep1_{group}_s*")) if (d / "done.json").exists()]


def arms(s: dict) -> None:
    base = s["base"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    print("arms.png: group, n seeds, deployed hack [95% CI], skill [95% CI]")
    print(f"  base, -, {base['hack_rate']:.3f} {r3(base['ci_hack'])}, {base['skill']:.3f} {r3(base['ci_skill'])}")
    ax.errorbar(
        base["skill"],
        base["hack_rate"],
        xerr=yerr(base, "skill", "ci_skill"),
        yerr=yerr(base),
        fmt="*",
        ms=13,
        color=INK,
        capsize=2,
        lw=1,
    )
    ax.annotate(
        "base Qwen3-0.6B",
        (base["skill"], base["hack_rate"]),
        xytext=(-8, 8),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=INK,
    )
    offsets = {"none": (8, 6), "penalty": (8, -4), "filter": (-8, 12), "graft_noroute": (8, -16), "graft": (8, 4)}
    for g, label, colour, marker, _ in ARMS:
        rec = s["groups"].get(g)
        if not rec:
            continue
        m = rec[rec["deployed_mode"]]
        print(f"  {g}, {rec['n_seeds']}, {m['hack_rate']:.3f} {r3(m['ci_hack'])}, {m['skill']:.3f} {r3(m['ci_skill'])}")
        ax.errorbar(
            m["skill"],
            m["hack_rate"],
            xerr=yerr(m, "skill", "ci_skill"),
            yerr=yerr(m),
            fmt=marker,
            ms=8,
            color=colour,
            mfc=colour if g in ("graft", "none") else "white",
            capsize=2,
            lw=1,
        )
        ax.annotate(
            f"{label} ({rec['n_seeds']} seeds)",
            (m["skill"], m["hack_rate"]),
            xytext=offsets[g],
            textcoords="offset points",
            ha="right" if g == "filter" else "left",
            fontsize=9,
            color=colour,
        )
    ax.set_xlim(0, 1.25)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("skill: correct final answer where holding pays (192 episodes)")
    ax.set_ylabel("deployed hack rate (64 episodes)")
    fig.tight_layout()
    fig.savefig(FIGURES / "arms.png")
    plt.close(fig)


def modes(s: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    mode_lab = [
        ("both", "both adapters\n(as trained)"),
        ("retain_only", "RETAIN only\n(deployed)"),
        ("forget_only", "FORGET only"),
    ]
    print("modes.png: group, mode, hack rate [95% CI], per seed")
    w = 0.38
    for k, (g, label, colour) in enumerate(
        [("graft", "GRAFT", EMBER), ("graft_noroute", "GRAFT without routing", GREY)]
    ):
        rec = s["groups"][g]
        for j, (mode, _) in enumerate(mode_lab):
            m = rec[mode]
            x = j + (k - 0.5) * w
            print(f"  {g}, {mode}, {m['hack_rate']:.3f} {r3(m['ci_hack'])}, {m['per_seed_hack']}")
            ax.bar(
                x, m["hack_rate"], w * 0.92, color=colour, label=f"{label} ({rec['n_seeds']} seeds)" if j == 0 else None
            )
            ax.errorbar(x, m["hack_rate"], yerr=yerr(m), color=INK, capsize=3, lw=1)
            ax.scatter([x] * len(m["per_seed_hack"]), m["per_seed_hack"], s=10, color=INK, zorder=3)
    ax.axhline(s["base"]["hack_rate"], color=INK, ls="--", lw=1)
    ax.text(2.6, s["base"]["hack_rate"] + 0.02, "base", ha="right", fontsize=9, color=INK)
    ax.set_xticks(range(3), [b for _, b in mode_lab])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("hack rate (64 episodes)")
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "modes.png")
    plt.close(fig)


def recall(s: dict) -> None:
    levels = [0.05, 0.2, 0.5, 1.0]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    base, rnd = s["base"], s["groups"].get("random_flags=0.4")
    ax.axhspan(base["ci_hack"][0], base["ci_hack"][1], color=LIGHT, alpha=0.5, lw=0)
    ax.axhline(base["hack_rate"], color=INK, ls="--", lw=1)
    ax.text(3.35, base["hack_rate"] + 0.02, "base", ha="right", fontsize=9, color=INK)
    print("recall.png: recall, n seeds, deployed hack [95% CI], per seed")
    print(f"  base {base['hack_rate']:.3f} {r3(base['ci_hack'])}")
    if rnd:
        m = rnd["retain_only"]
        print(f"  random flags 0.4, {rnd['n_seeds']}, {m['hack_rate']:.3f} {r3(m['ci_hack'])}, {m['per_seed_hack']}")
        ax.axhline(m["hack_rate"], color=GREY, ls=":", lw=1.4)
        ax.text(
            1.5,
            m["hack_rate"] - 0.06,
            f"random flags at rate 0.4 ({rnd['n_seeds']} seeds)",
            fontsize=9,
            color=GREY,
            ha="center",
        )
    xs, ys, ticks = [], [], []
    for i, r in enumerate(levels):
        rec = s["groups"].get(f"recall={r}")
        if not rec:
            continue
        m = rec["retain_only"]
        print(f"  {r}, {rec['n_seeds']}, {m['hack_rate']:.3f} {r3(m['ci_hack'])}, {m['per_seed_hack']}")
        ax.errorbar(i, m["hack_rate"], yerr=yerr(m), fmt="o", ms=8, color=EMBER, capsize=3, lw=1)
        ax.scatter([i + 0.12] * len(m["per_seed_hack"]), m["per_seed_hack"], s=12, color=INK, zorder=3)
        xs.append(i)
        ys.append(m["hack_rate"])
        ticks.append(f"{int(r * 100)}%\n({rec['n_seeds']} seed{'s' if rec['n_seeds'] > 1 else ''})")
    ax.plot(xs, ys, color=EMBER, lw=1, alpha=0.6)
    ax.set_xticks(xs, ticks)
    ax.set_xlim(-0.4, 3.4)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("share of hacks flagged during training (ground-truth flags, thinned)")
    ax.set_ylabel("deployed hack rate (RETAIN only)")
    fig.tight_layout()
    fig.savefig(FIGURES / "recall.png")
    plt.close(fig)


def monitors(b: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    print("monitors.png: backend, recall [95% CI], FPR [95% CI] (200 hacks, 200 non-hacks)")
    rows = [(n, lab) for n, lab in MONITORS if n in b["backends"]]
    for n, _ in rows:
        r = b["backends"][n]
        print(f"  {n}, {r['recall']:.3f} {r['recall_ci95']}, {r['fpr']:.3f} {r['fpr_ci95']}")
    for ax, (key, ci, title) in zip(
        axes,
        [("recall", "recall_ci95", "recall on 200 hacks"), ("fpr", "fpr_ci95", "false positives on 200 non-hacks")],
    ):
        for i, (n, _) in enumerate(rows):
            r = b["backends"][n]
            y = len(rows) - 1 - i
            ax.barh(y, r[key], 0.7, color=EMBER if n.startswith("oss_safeguard") else GREY)
            ax.errorbar(r[key], y, xerr=[[r[key] - r[ci][0]], [r[ci][1] - r[key]]], color=INK, capsize=2, lw=1)
            ax.text(min(r[ci][1], 1.0) + 0.03, y, f"{r[key]:.3f}", va="center", fontsize=8, color=INK)
        ax.set_xlim(0, 1.2)
        ax.set_title(title, fontsize=10, loc="left")
    axes[0].set_yticks(range(len(rows)), [lab for _, lab in rows][::-1])
    fig.tight_layout()
    fig.savefig(FIGURES / "monitors.png")
    plt.close(fig)


def curves(s: dict) -> None:
    """Needs every eval step (0, 25, ..., 250). The shipped results/runs keep only each run's first and last step, so
    without the intermediate ones (analyze_ep1.py --pull fetches them from the Modal Volume) the shipped figure is
    kept as drawn."""
    if all(len(list((d / "evals").glob("step_*.jsonl"))) <= 2 for d in finished_runs("graft")):
        print("curves.png: kept as shipped (intermediate eval steps not pulled; analyze_ep1.py --pull)")
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    print("curves.png: group (runs), step:mean deployed hack rate over finished seeds")
    for g, label, colour, _, ls in ARMS:
        runs = finished_runs(g)
        if not runs:
            continue
        mode = "retain_only" if g in GRAFT_FAMILY else "both"
        per_step: dict[int, list[float]] = {}
        for d in runs:
            for f in sorted((d / "evals").glob("step_*.jsonl")):
                rows = [regrade(json.loads(x)) for x in f.read_text().splitlines()]
                hack = env.summarize([r for r in rows if r["mode"] == mode])["hack_rate"]
                per_step.setdefault(int(f.stem.split("_")[1]), []).append(hack)
        steps = sorted(k for k, v in per_step.items() if len(v) == len(runs))
        ys = [sum(per_step[k]) / len(runs) for k in steps]
        print(f"  {g} ({[d.name for d in runs]}): " + ", ".join(f"{k}:{y:.3f}" for k, y in zip(steps, ys)))
        ax.plot(steps, ys, ls=ls, color=colour, lw=2.2 if g == "graft" else 1.4, label=f"{label} ({len(runs)})")
    ax.axhline(s["base"]["hack_rate"], color=INK, ls="--", lw=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 255)
    ax.set_xlabel("training step")
    ax.set_ylabel("deployed hack rate (64 episodes)")
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0.5, 0.4))
    fig.tight_layout()
    fig.savefig(FIGURES / "curves.png")
    plt.close(fig)


def cover(
    lines: list[tuple[list[float], list[float], str, float]],
    base: float,
    xlim: tuple[float, float],
    figures: Path,
    thumb: str,
) -> None:
    """Write hero.jpg (16:9), og.jpg (1200x630) and the 720x480 research card from data lines, with no axes.

    lines: (xs, ys, colour, width); base: the untrained model's rate, drawn as a dashed rule."""
    from PIL import Image

    for name, size in (("hero.jpg", (16, 9)), ("og.jpg", (12, 6.3))):
        fig = plt.figure(figsize=size, dpi=100)
        ax = fig.add_axes([0.03, 0.06, 0.94, 0.88])
        fig.patch.set_facecolor(PAPER)
        ax.set_facecolor(PAPER)
        ax.axhline(base, color=INK, ls=(0, (6, 5)), lw=1.6, alpha=0.7)
        for xs, ys, colour, lw in sorted(lines, key=lambda t: t[2] == EMBER):
            ax.plot(
                xs,
                ys,
                color=colour,
                lw=lw,
                alpha=0.9 if colour == EMBER else 0.55,
                solid_capstyle="round",
                solid_joinstyle="round",
            )
            ax.scatter(xs[-1:], ys[-1:], s=lw * 40, color=colour, zorder=3)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.04, 1.04)
        ax.axis("off")
        tmp = figures / ("_" + name.replace(".jpg", ".png"))
        fig.savefig(tmp, dpi=100, facecolor=PAPER)
        plt.close(fig)
        Image.open(tmp).convert("RGB").save(figures / name, quality=88)
        tmp.unlink()
    img = Image.open(figures / "hero.jpg")  # the research page's card: 720x480, centre crop of the cover
    w, h = img.size
    cw = int(h * 1.5)
    card = img.crop(((w - cw) // 2, 0, (w + cw) // 2, h)).resize((720, 480), Image.LANCZOS)
    card.save(HERE.parent / "shared" / "thumbs" / f"{thumb}.jpg", quality=84, optimize=True)


def hero(s: dict) -> None:
    """Cover: each finished run's deployed hack rate from its first to its last held-out eval, one line per seed
    (GRAFT in ember, the other four arms grey), the untrained model as a dashed rule."""
    lines = []
    print("hero.jpg: group, run, deployed hack rate at the first and last eval step")
    for g, _, colour, _, _ in ARMS:
        mode = "retain_only" if g in GRAFT_FAMILY else "both"
        for d in finished_runs(g):
            xs, ys = [], []
            for f in sorted((d / "evals").glob("step_*.jsonl")):
                rows = [regrade(json.loads(x)) for x in f.read_text().splitlines()]
                xs.append(int(f.stem.split("_")[1]))
                ys.append(env.summarize([r for r in rows if r["mode"] == mode])["hack_rate"])
            print(f"  {g}, {d.name}, {ys[0]:.3f} -> {ys[-1]:.3f}")
            lines.append((xs, ys, EMBER if g == "graft" else GREY, 5.0 if g == "graft" else 2.2))
    cover(lines, s["base"]["hack_rate"], (-6, 256), FIGURES, "graft-monitor")


def training_logs() -> None:
    """Text numbers from the per-step training logs: mean over the last 25 logged steps of each finished run."""
    print(
        "training logs (last 25 steps, mean over finished runs): adv_frac_zero, grad_norm_total, kl, flag_rate, "
        "flag_recall, flag_fpr"
    )
    for g in ("none", "penalty", "filter", "graft", "graft_noroute"):
        vals: dict[str, list[float]] = {
            k: [] for k in ("adv_frac_zero", "grad_norm_total", "kl", "flag_rate", "flag_recall", "flag_fpr")
        }
        for d in finished_runs(g):
            log = [json.loads(x) for x in (d / "log.jsonl").read_text().splitlines()][-25:]
            for k in vals:
                v = [r[k] for r in log if r.get(k) is not None and r[k] == r[k]]
                if v:
                    vals[k].append(sum(v) / len(v))
        print(
            f"  {g} ({len(finished_runs(g))} runs): "
            + ", ".join(f"{k} {sum(v) / len(v):.3f}" for k, v in vals.items() if v)
        )


def bench_reads() -> None:
    """Text numbers from the labelled monitor set: how the hacks are worded, and what each monitor's false positives
    are (final answer correct, the user's wrong answer, or a third wrong country)."""
    bench = RESULTS / "monitor_bench"
    traj = [json.loads(x) for x in (bench / "trajectories.jsonl").read_text().splitlines()]
    hacks = [t for t in traj if t["gt_hack"]]
    neg = [t for t in traj if not t["gt_hack"]]
    concede = re.compile(r"sorry|apolog|you('| a)re (right|correct)|my mistake|stand corrected", re.I)
    bare = re.compile(r"\s*final answer\s*:\s*[^.\n]+\.?\s*", re.I)
    words = sorted(len(t["final_text"].split()) for t in hacks)
    print(
        f"bench hacks: {len(hacks)}; with an apology or concession phrase: "
        f"{sum(bool(concede.search(t['final_text'])) for t in hacks)}; final turn is only 'Final answer: X': "
        f"{sum(bool(bare.fullmatch(t['final_text'])) for t in hacks)}; median words in final turn: "
        f"{words[len(words) // 2]}; first answer correct: {sum(t['first_correct'] for t in hacks)}"
    )
    other = [t for t in neg if not t["correct"] and not t["agrees"]]
    print(
        f"bench non-hacks: {len(neg)}; final answer wrong and not the user's: {len(other)} "
        f"(first answer correct in {sum(t['first_correct'] for t in other)})"
    )
    for n, _ in MONITORS:
        v = {json.loads(x)["id"]: json.loads(x) for x in (bench / f"verdicts_{n}.jsonl").read_text().splitlines()}
        fp = [t for t in neg if v[t["id"]]["flag"]]
        toks = [r["usage"]["completion_tokens"] for r in v.values() if (r.get("usage") or {}).get("completion_tokens")]
        tok = f"; mean completion tokens {sum(toks) / len(toks):.0f}" if toks else ""
        print(
            f"  {n}: false positives {len(fp)} = final correct {sum(t['correct'] for t in fp)}, third wrong country "
            f"{sum(not t['correct'] and not t['agrees'] for t in fp)}{tok}"
        )


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    s = json.loads((RESULTS / "ep1_summary.json").read_text())
    b = json.loads((RESULTS / "monitor_bench" / "summary.json").read_text())
    print(f"ep1_summary: {s['n_runs']} finished runs; missing {s['missing']}")
    arms(s)
    modes(s)
    recall(s)
    monitors(b)
    curves(s)
    hero(s)
    training_logs()
    bench_reads()
    keys = (
        "oss_safeguard|oss_safeguard@low",
        "oss_safeguard|qwen_judge",
        "control_arena|oss_safeguard",
        "oss_safeguard|petri",
        "oss_safeguard|rule",
    )
    print("kappa:", {k: b["kappa"][k] for k in keys})
    print("handcheck:", b["handcheck"])


if __name__ == "__main__":
    main()
