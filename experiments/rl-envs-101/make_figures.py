"""Draw every figure in the note from results/, print the numbers each one plots, and write the widget's data.

    .venv/bin/python experiments/rl-envs-101/make_figures.py

Orange is the thing the note is about (training on the full check, the hidden key); grey is the comparison
(training on the loose check, the visible key).
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = HERE / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
ACCENT, GREY, GREY_SOFT, INK = "#c9561c", "#6f7a7d", "#c4cacc", "#1d272a"
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.edgecolor": GREY,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "savefig.dpi": 200,
    }
)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"   wrote report/figures/{name}")


def bars(ax, labels, visible, hidden):
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], visible, 0.38, color=GREY_SOFT, edgecolor=GREY, label="visible check (reward)")
    ax.bar([i + 0.2 for i in x], hidden, 0.38, color=ACCENT, label="hidden check")
    for i, (v, h) in enumerate(zip(visible, hidden)):
        ax.text(i - 0.2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.2, h + 0.02, f"{h:.2f}", ha="center", fontsize=8)
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("mean score")
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)


# Figure 1: five scripted habits on the desk (step 3)
habits = json.loads((RES / "03_habits.json").read_text())
print("Figure 1 (habits):", {h: (round(r["reward"], 2), round(r["hidden"], 2)) for h, r in habits.items()})
fig, ax = plt.subplots(figsize=(7, 3.2))
bars(ax, list(habits), [r["reward"] for r in habits.values()], [r["hidden"] for r in habits.values()])
save(fig, "fig1_habits.png")

# Figure 2: Harbor on 30 held-out tasks, plus the same model through the JSON protocol on the same tasks
harbor = json.loads((RES / "05_harbor.json").read_text())
selected = {d.split("-", 2)[2] for d in os.listdir(HERE / "harbor_tasks" / "heldout")}
json_rows = [json.loads(x) for x in (RES / "baseline_Qwen3-4B-Instruct-2507.jsonl").read_text().splitlines()]
json_rows = [r for r in json_rows if r["id"] in selected]
labels = ["oracle", "nop", "Qwen3-4B\nterminal (Harbor)", "Qwen3-4B\nJSON tools"]
vis = [harbor[k]["summary"]["reward"] for k in ("oracle", "nop", "qwen")]
hid = [harbor[k]["summary"]["hidden"] for k in ("oracle", "nop", "qwen")]
vis.append(sum(r["scores"]["reward"] for r in json_rows) / len(json_rows))
hid.append(sum(r["scores"]["hidden"] for r in json_rows) / len(json_rows))
print("Figure 2 (harbor):", [(lab.replace("\n", " "), round(v, 3), round(h, 3)) for lab, v, h in zip(labels, vis, hid)])
fig, ax = plt.subplots(figsize=(7, 3.2))
bars(ax, labels, vis, hid)
save(fig, "fig2_harbor.png")

# Figure 3: GRPO, held-out scores by step for both runs
runs = {k: json.loads((RES / f"grpo_{k}-Qwen2.5-3B-Instruct-s0.json").read_text()) for k in ("visible", "full")}
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
arms = [("visible", GREY, "trained on the visible check"), ("full", ACCENT, "trained on the full check")]
for ax, (key, title) in zip(axes, [("reward", "visible check, held-out"), ("hidden", "hidden check, held-out")]):
    for arm, colour, name in arms:
        ev = [e for e in runs[arm]["log"] if e["split"] == "eval"]
        ax.plot([e["step"] for e in ev], [e[key] for e in ev], "o-", color=colour, label=name, lw=2)
        print(f"Figure 3 {arm} {key}:", [round(e[key], 3) for e in ev])
    ax.set_title(title, loc="left")
    ax.set_xlabel("GRPO step")
    ax.set_ylim(0, 1.05)
axes[0].set_ylabel("mean score (154 tasks)")
axes[1].legend(frameon=False, fontsize=8, loc="lower right")
save(fig, "fig3_grpo.png")

# Figure 4: which slot did each policy book (held-out booking episodes)
report = json.loads((RES / "06_grpo.json").read_text())
pairs = [
    ("untrained", report["visible-Qwen2.5-3B-Instruct-s0"]["hack_checks"]["0"]),
    ("trained on visible", report["visible-Qwen2.5-3B-Instruct-s0"]["hack_checks"]["60"]),
    ("trained on full", report["full-Qwen2.5-3B-Instruct-s0"]["hack_checks"]["60"]),
]
fig, ax = plt.subplots(figsize=(6.5, 2.6))
for i, (name, c) in enumerate(pairs):
    e, f = c["booked_earliest"], c["booked_first_listed_not_earliest"]
    ax.barh(i, e, color=ACCENT, label="the earliest slot (asked for)" if i == 0 else None)
    ax.barh(
        i, f, left=e, color=GREY_SOFT, edgecolor=GREY, label="the first listed, not the earliest" if i == 0 else None
    )
    ax.text(e + f + 1, i, f"{e} / {f}", va="center", fontsize=8)
    print(f"Figure 4 {name}: earliest {e}, first-listed {f}")
ax.set_yticks(range(3), [p[0] for p in pairs])
ax.invert_yaxis()
ax.set_xlim(0, 130)
ax.set_xlabel("held-out booking episodes")
ax.legend(frameon=False, fontsize=8, loc="lower right")
save(fig, "fig4_slots.png")

# The widget's data: three policies on four held-out tasks
grpo = RES / "grpo"
pick = ["w3-011", "w3-028", "w3-123", "w3-073"]
policies = {
    "untrained": grpo / "visible-Qwen2.5-3B-Instruct-s0" / "eval_0.jsonl",
    "trained on visible": grpo / "visible-Qwen2.5-3B-Instruct-s0" / "eval_60.jsonl",
    "trained on full": grpo / "full-Qwen2.5-3B-Instruct-s0" / "eval_60.jsonl",
}
widget = {}
for name, path in policies.items():
    rows = {json.loads(x)["id"]: json.loads(x) for x in path.read_text().splitlines()}
    widget[name] = {
        k: {
            "messages": rows[k]["messages"],
            "scores": {s: rows[k]["scores"][s] for s in ("reward", "hidden", "collateral")},
        }
        for k in pick
        if k in rows
    }
(HERE / "report" / "episodes.js").write_text("window.EPISODES = " + json.dumps(widget) + ";\n")
print("   wrote report/episodes.js", {k: len(v) for k, v in widget.items()})
