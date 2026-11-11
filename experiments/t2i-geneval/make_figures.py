"""Draw the harness note's figures from results/, printing every number plotted.

  python make_figures.py              figures from the result files
  python make_figures.py --gallery    also fetch the 50 blind-look pairs from the Modal volume (report/figures/pairs/)

gains.png    GenEval on the official 553 prompts per category: FLUX.1-schnell alone, the planner alone (1 image), the
             harness; 95% bootstrap intervals over prompts.
ablation.png validation: GenEval overall against images drawn per output, for each prompt variant and picker.
rivals.png   published GenEval overall scores beside ours, with images drawn per output.
guards.png   GenAI-Bench VQAScore (planner, planner + PickScore best of 4) and T2I-CompBench B-VQA (planner), paired
             differences from the original prompt; the blind look's counts.
style.jpg    six blind-look pairs, FLUX.1-schnell alone above the harness.
viewer       the pair viewer's data, written into report/index.html between the viewer:start and viewer:end
             markers.
"""

import csv
import io
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import HERE, RESULTS, TAG_ORDER, geneval_summary, read_jsonl
from pilot_summary import PRIMARY, compbench, harness_rows

FIG = HERE / "report" / "figures"
EMBER, GREY, LIGHT, INK = "#C9561C", "#6F6A63", "#C8C2B8", "#2B2723"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "savefig.dpi": 150})  # fmt: skip
TAG_LABEL = {"single_object": "Single object", "two_object": "Two objects", "counting": "Counting",
             "colors": "Colours", "position": "Position", "color_attr": "Colour binding",
             "overall": "Overall"}  # fmt: skip


def gains():
    base = geneval_summary(read_jsonl(RESULTS / "schnell_official/rows.jsonl"))
    plan1 = harness_rows("harness_official", "plan | single | N=1")[1]["geneval"]
    har = harness_rows("harness_official", PRIMARY)[1]["geneval"]
    keys = [*TAG_ORDER, "overall"]
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    x = np.arange(len(keys))
    print("gains.png: category, base, planner 1 image, harness (rate [lo, hi])")
    series = ((base, GREY, "FLUX.1-schnell alone"), (plan1, LIGHT, "planner only, 1 image"), (har, EMBER, "harness"))
    for off, (s, col, lab) in zip((-0.27, 0, 0.27), series, strict=True):
        v = np.array([s[k][0] for k in keys])
        lo = v - np.array([s[k][1] for k in keys])
        hi = np.array([s[k][2] for k in keys]) - v
        ax.bar(x + off, v, 0.26, color=col, label=lab, yerr=[lo, hi], error_kw={"lw": 0.8, "ecolor": INK})
    for k in keys:
        print(f"  {k:<14} " + "  ".join(f"{s[k][0]:.3f} [{s[k][1]:.3f}, {s[k][2]:.3f}]" for s, _, _ in series))
    ax.set_xticks(x, [TAG_LABEL[k] for k in keys])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("GenEval pass rate")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "gains.png")
    plt.close(fig)


def ablation():
    table = json.loads((RESULTS / "harness_val/ablation.json").read_text())
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    print("ablation.png: config, candidates per output, overall")
    styles = {"random": (LIGHT, "-"), "pickscore": (GREY, "-"), "indep": (EMBER, "-"), "oracle": (EMBER, ":")}
    names = {"random": "random pick", "pickscore": "PickScore picks", "indep": "checker picks",
             "oracle": "official grader picks (bound)"}  # fmt: skip
    for variant, marker in (("orig", "s"), ("plan", "o")):
        single = next(t for t in table if t["variant"] == variant and t["picker"] == "single")
        for picker, (col, ls) in styles.items():
            more = [t for t in table if t["variant"] == variant and t["picker"] == picker and not t["retry"]]
            pts = [single, *sorted(more, key=lambda t: t["n"])]
            xs = [t["candidates_per_output"] for t in pts]
            ys = [t["geneval"]["overall"][0] for t in pts]
            lab = f"{'planner' if variant == 'plan' else 'original'}, {names[picker]}"
            ax.plot(xs, ys, ls=ls, marker=marker, color=col, alpha=1 if variant == "plan" else 0.5, label=lab, ms=4)
            print(f"  {lab}: " + ", ".join(f"N={t['n']} {y:.3f}" for t, y in zip(pts, ys, strict=True)))
    r = next(t for t in table if t["config"] == PRIMARY)
    ax.plot(r["candidates_per_output"], r["geneval"]["overall"][0], marker="*", ms=13, color=EMBER, ls="none",
            label="harness: planner, checker, retry")  # fmt: skip
    print(f"  {PRIMARY}: {r['candidates_per_output']:.2f} images, {r['geneval']['overall'][0]:.3f}")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("images drawn per output (log scale)")
    ax.set_ylabel("GenEval overall, validation\n(axis starts at 0.5)")
    ax.legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(FIG / "ablation.png")
    plt.close(fig)


def rivals():
    with open(HERE / "data/rivals.csv") as f:
        rows = list(csv.DictReader(f))
    keep, seen = [], set()
    for r in rows:
        if r["model"] != "FLUX.1-schnell" and r["model"] not in seen:  # our own base run is shown instead
            seen.add(r["model"])
            keep.append(r)
    har = harness_rows("harness_official", PRIMARY)[1]
    base = geneval_summary(read_jsonl(RESULTS / "schnell_official/rows.jsonl"))
    pts = [(r["model"], float(r["overall"]), r["kind"], r["samples_per_output"]) for r in keep]
    spo = f"{har['candidates_per_output']:.1f}"
    pts += [("FLUX.1-schnell + this harness", har["geneval"]["overall"][0], "ours", spo),
            ("FLUX.1-schnell alone (our run)", base["overall"][0], "base", "1")]  # fmt: skip
    pts.sort(key=lambda p: p[1])
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    print("rivals.png: system, overall, kind, images per output")
    for y, (name, v, kind, spo) in enumerate(pts):
        single = kind in ("open model", "closed model", "base")
        col = EMBER if kind == "ours" else (GREY if single else INK)
        mk = "*" if kind == "ours" else ("o" if single else "D")
        ax.plot(v, y, marker=mk, color=col, ms=12 if kind == "ours" else 6, ls="none")
        if kind == "ours":
            ax.plot(har["geneval"]["overall"][1:3], [y, y], color=EMBER, lw=2)
        label = name + (f"  ({spo} images per output)" if spo not in ("", "1") else "")
        ax.text(0.505, y, label, va="center", fontsize=8.2, color=EMBER if kind == "ours" else INK)
        print(f"  {name}: {v:.3f} ({kind}, {spo or '?'} per output)")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("GenEval overall (axis starts at 0.5)")
    fig.tight_layout()
    fig.savefig(FIG / "rivals.png")
    plt.close(fig)


def guards():
    g = json.loads((RESULTS / "genai_bench/summary.json").read_text())
    cb = compbench("cb_base1024", "cb_plan1024")
    blind = json.loads((RESULTS / "blind/summary.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3), gridspec_kw={"width_ratios": [1.1, 1.3, 1.2]})
    print("guards.png")

    def dots(ax, items):
        for y, (lab, (m, lo, hi)) in enumerate(items):
            ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="o", color=EMBER if hi < 0 or lo > 0 else GREY, capsize=3)
            print(f"  {lab.replace(chr(10), ' ')}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_yticks(range(len(items)), [i[0] for i in items], fontsize=8)
        ax.set_ylim(-0.6, len(items) - 0.4)
        ax.set_xlabel("change vs original prompt")

    dots(axes[0], [("planner, 1 image", g["planner"]["vqa_diff"]),
                   ("planner + PickScore\nbest of 4", g["planner_pick4"]["vqa_diff"])])  # fmt: skip
    axes[0].set_title(f"GenAI-Bench, VQAScore\n({g['prompts']} prompts; base {g['base']['vqa']:.3f})", fontsize=9)
    dots(axes[1], [(c.capitalize().replace("Color", "Colour"), cb[c][2:]) for c in ("color", "shape", "texture")])
    axes[1].set_title("T2I-CompBench, B-VQA\n(100 prompts each; planner, 1 image)", fontsize=9)
    ax = axes[2]
    for y, q in enumerate(("natural", "match")):
        h, t, b = blind[q]["harness"], blind[q]["tie"], blind[q]["base"]
        for x0, w, col in ((0, h, EMBER), (h, t, LIGHT), (h + t, b, GREY)):
            ax.barh(y, w, left=x0, color=col)
            if w:
                ax.text(x0 + w / 2, y, str(w), ha="center", va="center", fontsize=8, color="white" if w > 3 else INK)
        print(f"  blind look {q}: harness {h}, tie {t}, base {b}")
    ax.set_yticks([0, 1], ["Looks more\nnatural", "Shows the\nprompt"], fontsize=8)
    ax.set_xlim(0, blind["pairs"])
    ax.set_title(f"Blind look, {blind['pairs']} pairs\nharness | tie | base", fontsize=9)
    ax.set_xlabel("pairs")
    fig.tight_layout()
    fig.savefig(FIG / "guards.png")
    plt.close(fig)


def gallery():
    """The blind-look pairs as small JPEGs for the pair viewer and the style figure."""
    import modal
    from PIL import Image

    key = json.loads((RESULTS / "blind/key.json").read_text())
    imgs = modal.Function.from_name("t2i-geneval", "fetch").remote([p for k in key for p in k["paths"].values()], 384)
    out = FIG / "pairs"
    out.mkdir(parents=True, exist_ok=True)
    for k in key:
        for side, p in k["paths"].items():
            Image.open(io.BytesIO(imgs[p])).convert("RGB").save(out / f"{k['pair']:02d}_{side}.jpg", quality=82)
    print(f"gallery: {len(key)} pairs written")


def style():
    """The first blind-look pair of each category, FLUX.1-schnell alone above the harness."""
    from PIL import Image

    key = json.loads((RESULTS / "blind/key.json").read_text())
    picks = [next(k["pair"] for k in key if k["tag"] == t) for t in TAG_ORDER]  # the first pair of each category
    w = 300
    sheet = Image.new("RGB", (len(picks) * (w + 8) - 8, 2 * w + 8), "white")
    for c, n in enumerate(picks):
        for r, side in enumerate(("base", "harness")):
            im = Image.open(FIG / "pairs" / f"{n:02d}_{side}.jpg").resize((w, w), Image.LANCZOS)
            sheet.paste(im, (c * (w + 8), r * (w + 8)))
    sheet.save(FIG / "style.jpg", quality=85)
    print("style.jpg: pairs", picks)


def viewer():
    """The 50 blind-look pairs with prompt, planner rewrite and both grader verdicts, as JSON inside the note."""
    key = json.loads((RESULTS / "blind/key.json").read_text())
    plans = {r["key"]: r["plan"] for r in read_jsonl(RESULTS / "plans/official.jsonl")}
    replans = {r["key"]: r["plan"] for r in read_jsonl(RESULTS / "plans/official_retry.jsonl")}
    pool = {r["path"]: r for r in read_jsonl(RESULTS / "harness_official/pool.jsonl")}
    base = {r["path"]: r for r in read_jsonl(RESULTS / "schnell_official/rows.jsonl")}
    ratings = {r["pair"]: r for r in json.loads((RESULTS / "blind/ratings.json").read_text())}
    data = []
    for k in key:
        h = pool[k["paths"]["harness"]]
        retry = h["variant"] == "retry"
        r = ratings[k["pair"]]
        data.append({
            "n": k["pair"], "tag": TAG_LABEL[k["tag"]], "prompt": k["prompt"],
            "plan": replans[f"{k['idx']}_{h['rep']}"] if retry else plans[str(k["idx"])], "retry": retry,
            "base_ok": bool(base[k["paths"]["base"]]["correct"]), "har_ok": bool(h["correct"]),
            "match": {"A": side_name(k, "A"), "B": side_name(k, "B")}.get(r["match"], "tie"),
            "natural": {"A": side_name(k, "A"), "B": side_name(k, "B")}.get(r["natural"], "tie"),
        })  # fmt: skip
    block = (
        '<!-- viewer:start -->\n<script type="application/json" id="pairs-data">'
        + json.dumps(data).replace("</", "<\\/")
        + "</script>\n<!-- viewer:end -->"
    )
    page = HERE / "report" / "index.html"
    text = page.read_text()
    a, b = text.index("<!-- viewer:start -->"), text.index("<!-- viewer:end -->") + len("<!-- viewer:end -->")
    page.write_text(text[:a] + block + text[b:])
    print(f"viewer: {len(data)} pairs written into report/index.html")


def side_name(k, side):
    return "harness" if k[side] == "harness" else "base"


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    if "--gallery" in sys.argv:
        gallery()
    gains()
    ablation()
    rivals()
    guards()
    style()
    if (HERE / "report" / "index.html").exists():
        viewer()
