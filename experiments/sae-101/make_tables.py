"""
Write the note's data tables from results/<size>/*.json into report/index.html, between `<!-- table:NAME:start -->`
and `<!-- table:NAME:end -->`, so no table number is typed by hand. Rerun after any result changes.

    .venv/bin/python experiments/sae-101/make_tables.py --size full
"""

import argparse
import html
import re

import sae101 as s

ap = argparse.ArgumentParser()
ap.add_argument("--size", choices=list(s.SIZES), default="full")
args = ap.parse_args()
res = s.RESULTS / args.size
ev, it, cp = s.read_json(res / "evaluate.json"), s.read_json(res / "interpret.json"), s.read_json(res / "compare.json")
al, mk, tr = (
    s.read_json(res / "autolabels.json"),
    s.read_json(res / "mistakes.json"),
    s.read_json(res / "train_handwritten.json"),
)
trl = s.read_json(res / "train_saelens.json")
hw, sl = ev["saes"]["hand-written"], ev["saes"].get("SAELens")
e = html.escape
NUM = ' class="num"'
HL = ' class="hl"'


def wrap(head, rows, cls=""):
    th = "".join(f"<th{NUM if n else ''}>{h}</th>" for h, n in head)
    lines = []
    for cells, hl in rows:
        tds = "".join(f"<td{NUM if head[i][1] else ''}>{c}</td>" for i, c in enumerate(cells))
        lines.append(f"        <tr{HL if hl else ''}>{tds}</tr>")
    body = "\n".join(lines)
    return f'  <div class="table-wrap{(" " + cls) if cls else ""}">\n    <table>\n      <thead><tr>{th}</tr></thead>\n      <tbody>\n{body}\n      </tbody>\n    </table>\n  </div>'


def ci(v, digits=3):
    return f"[{v[0]:.{digits}f}, {v[1]:.{digits}f}]"


def verdict(ok):
    return "pass" if ok else '<span class="fail">miss</span>'


tables = {}

# ---- evaluation, against the pass bars ----
rows = []
for label, bar, fmt, ok in (
    ("FVU (held-out)", "&lt; 0.25", lambda r: f"{r['fvu']:.3f}", lambda r: r["fvu"] < 0.25),
    ("L0 (latents on per token)", "= k", lambda r: f"{r['l0']:.2f}", lambda r: abs(r["l0"] - r["k"]) < 0.01),
    (
        "Dead latents (never fire)",
        "&lt; 10%",
        lambda r: f"{r['dead']:,} ({r['dead_share']:.1%})",
        lambda r: r["dead_share"] < 0.10,
    ),
    (
        "Dense latents (&gt; 1 token in 10)",
        "&lt; 1%",
        lambda r: f"{r['dense']:,} ({r['dense_share']:.2%})",
        lambda r: r["dense_share"] < 0.01,
    ),
    ("Cross-entropy with the SAE (nats)", "—", lambda r: f"{r['ce']['sae']:.3f} (+{r['delta_ce_sae']:.3f})", None),
    (
        "CE recovered vs mean-ablation",
        "≥ 0.90",
        lambda r: f"{r['ce_recovered_vs_mean']:.3f} {ci(r['ce_recovered_vs_mean_ci'])}",
        lambda r: r["ce_recovered_vs_mean"] >= 0.90,
    ),
    (
        "Its error vs a random error of the same size",
        "—",
        lambda r: f"{r['sae_vs_random_ratio']:.1f}x {ci(r['sae_vs_random_ratio_ci'], 1)}",
        None,
    ),
):
    cells = [label, bar, fmt(hw) + (f" · {verdict(ok(hw))}" if ok else "")]
    if sl:
        cells.append(fmt(sl) + (f" · {verdict(ok(sl))}" if ok else ""))
    rows.append((cells, False))
head = [("Measure", False), ("Bar", False), ("Hand-written", True)] + ([("SAELens", True)] if sl else [])
tables["eval"] = wrap(head, rows, "t-eval")

# ---- the two trainings side by side ----
if trl and sl:
    rows = [
        (
            ["Code", "about 60 lines (<code>sae101.py</code>)", "a library call (<code>03b_train_saelens.py</code>)"],
            False,
        ),
        (["Same windows, layer, d_sae, k, learning rate, batch, tokens", "yes", "yes"], False),
        (["Minutes on one A10G", f"{tr['minutes']:.0f}", f"{trl['minutes']:.0f}"], False),
        (["Held-out FVU", f"{hw['fvu']:.3f}", f"{sl['fvu']:.3f}"], True),
        (["Dead latents", f"{hw['dead']:,}", f"{sl['dead']:,}"], False),
        (
            ["CE recovered vs mean-ablation", f"{hw['ce_recovered_vs_mean']:.3f}", f"{sl['ce_recovered_vs_mean']:.3f}"],
            False,
        ),
    ]
    tables["train"] = wrap([("", False), ("Hand-written", True), ("SAELens 6.56.1", True)], rows)

# ---- the drawn latents ----
labels = {r["id"]: r for r in (al or {}).get("latents", [])}
use = {u["id"]: u for u in it["use_test"]["per_latent"]}


def snippet(sn):
    out = []
    for i, t in enumerate(sn["tokens"]):
        t = e(t.replace("\n", "⏎"))
        out.append(f"<mark>{t}</mark>" if i == sn["at"] else t)
    return '<span class="snip">' + "".join(out).strip() + "</span>"


rows = []
for lat in it["latents"][: it["n_note"]]:
    lab = labels.get(lat["id"])
    u = use.get(lat["id"])
    check = f"{lab['accuracy'] * 10:.0f}/10 vs {lab['accuracy_wrong_label'] * 10:.0f}/10" if lab else "—"
    rows.append(
        (
            [
                f'#{lat["id"]}<br><span class="sub">{lat["freq"]:.2%} of tokens</span>',
                snippet(lat["top"][0]),
                e(lab["label"]) if lab else "—",
                f'{check}<br><span class="sub">use {u["kl_feature"] / u["kl_random"]:.1f}x</span>' if u else check,
            ],
            False,
        )
    )
tables["latents"] = wrap(
    [
        ("Latent", False),
        ("Strongest example (firing token marked)", False),
        ("Automatic label", False),
        ("Label check · use test", True),
    ],
    rows,
    "t-latents",
)

# ---- SAE vs probe vs diff of means vs PCA ----
methods = [
    ("probe", "Logistic probe"),
    ("diff_means", "Difference of means"),
    ("sae_latent", "Best single SAE latent"),
    ("sae_top5_probe", "Probe on the 5 best SAE latents"),
    ("pca_1", "PCA, first component"),
    ("pca_best10", "PCA, best of the first 10"),
    ("leace_probe", "Fresh probe after LEACE erasure"),
]
rows = []
for key, label in methods:
    cells = [label]
    for concept in ("code", "digit"):
        v, c = cp[concept][key]
        extra = ""
        if key == "sae_latent":
            extra = f" (#{cp[concept]['sae_latent_id']})"
        elif key == "pca_best10":
            extra = f" (PC{cp[concept]['pca_best_index']})"
        cells.append(f"{v:.3f} {ci(c)}{extra}")
    rows.append((cells, key == "sae_latent"))
rows.append(
    (
        ["100 random directions: mean, max"]
        + [f"{cp[c]['random']['mean']:.3f}, {cp[c]['random']['max']:.3f}" for c in ("code", "digit")],
        False,
    )
)
tables["compare"] = wrap(
    [
        ("Tool", False),
        (f"Code vs prose ({cp['code']['n_test']:,} held-out tokens)", True),
        (f"Digit tokens ({cp['digit']['n_test']:,})", True),
    ],
    rows,
)

# ---- common mistakes ----
if mk:
    base = mk["base"]
    rows = [(["The right recipe (TopK, k = 32)", f"{base['fvu']:.3f}", f"{base['dead_share']:.1%}", "—"], True)]
    why = {
        "no AuxK": "dead latents never get a gradient back",
        "learning rate 10x too high (4e-3)": "the steps overshoot",
        "learning rate 10x too low (4e-5)": "too slow for the token budget",
        "no normalisation (raw activations)": "the learning rate means something else at this scale",
        "attention sinks left in": "a few huge tokens dominate the error",
        "1/20 of the tokens": "the dictionary has not settled",
    }
    for r in mk["training"]:
        rows.append(
            (
                [
                    e(r["name"][0].upper() + r["name"][1:]),
                    f"{r['fvu']:.3f}",
                    f"{r['dead_share']:.1%}",
                    why.get(r["name"], ""),
                ],
                False,
            )
        )
    right = mk["usage"].get("right layer (block 11 output)")
    for name, v in mk["usage"].items():
        if name.startswith("right"):
            continue
        rows.append(
            (
                [
                    e(name[0].upper() + name[1:]) + " (the main SAE)",
                    f"{v:.3f}",
                    "—",
                    f"used on the wrong numbers: {v / right:.1f}x the error of the right use ({right:.3f})",
                ],
                False,
            )
        )
    tables["mistakes"] = wrap(
        [("Mistake", False), ("Held-out FVU", True), ("Never fired", True), ("Why it hurts", False)], rows
    )

index = s.HERE / "report" / "index.html"
page = index.read_text()
for name, body in tables.items():
    pattern = re.compile(rf"(<!-- table:{name}:start -->)(.*?)(<!-- table:{name}:end -->)", re.DOTALL)
    if not pattern.search(page):
        print(f"   (no table:{name} markers in the page)")
        continue
    page = pattern.sub(lambda m, b=body: m.group(1) + "\n" + b + "\n  " + m.group(3), page)
    print(f"   wrote table:{name}")
index.write_text(page)
