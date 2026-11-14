"""Self-contained HTML pages for the three tools' results, in steerlab's ember-and-cream style.

Each `render_*` returns one HTML string with inline CSS and no scripts beyond a theme toggle, so a report can be
attached to an issue, opened offline, or served as a static file. `save_html` writes it.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from pathlib import Path

from circuitkit.diff.core import DiffReport, UnitChange
from circuitkit.labels.autointerp import Calibration, FeatureLabel
from circuitkit.labels.neuronpedia import Comparison
from circuitkit.validate.types import VERDICT_MEANING, Report, Verdict

CSS = """
:root{--canvas:#faf7f2;--ink:#1d272a;--soft:#586468;--faint:#8a9497;--rule:#e6e0d6;--panel:#f3eee6;
--accent:#c9561c;--accent-soft:#f8e6dc;--control:#4a86e8;--good:#2f7d4f;--ground:#15110e}
:root[data-theme=dark]{--canvas:#15110e;--ink:#e9e3db;--soft:#b3aaa0;--faint:#8c847b;--rule:#2c251f;
--panel:#1f1914;--accent:#e07a3f;--accent-soft:#3a2419;--control:#6f9ff0;--good:#5fb27f}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--canvas:#15110e;--ink:#e9e3db;--soft:#b3aaa0;
--faint:#8c847b;--rule:#2c251f;--panel:#1f1914;--accent:#e07a3f;--accent-soft:#3a2419;--control:#6f9ff0;--good:#5fb27f}}
*{box-sizing:border-box}body{margin:0;background:var(--canvas);color:var(--ink);
font:16px/1.55 ui-serif,Georgia,serif}main{max-width:980px;margin:0 auto;padding:28px 16px 64px}
header.band{background:var(--ground);color:#d9d2ca;padding:22px 16px}header.band div{max-width:980px;margin:0 auto}
header.band h1{font:600 26px/1.2 ui-sans-serif,system-ui,sans-serif;margin:0 0 4px}
header.band p{margin:0;color:#a39a90;font:14px ui-sans-serif,system-ui,sans-serif}
h2{font:600 19px ui-sans-serif,system-ui,sans-serif;margin:34px 0 8px;border-bottom:1px solid var(--rule);
padding-bottom:4px}h3{font:600 16px ui-sans-serif,system-ui,sans-serif;margin:22px 0 6px}
table{border-collapse:collapse;width:100%;font:13.5px ui-sans-serif,system-ui,sans-serif;margin:8px 0}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--rule);vertical-align:top}
th{color:var(--soft);font-weight:600}td.n{text-align:right;font-variant-numeric:tabular-nums}
.wrap{overflow-x:auto}.pill{display:inline-block;padding:1px 9px;border-radius:99px;font:600 12px
ui-sans-serif,system-ui,sans-serif;background:var(--panel);color:var(--soft)}
.pill.switch,.pill.gained,.pill.up,.pill.ours{background:var(--accent-soft);color:var(--accent)}
.pill.readout,.pill.lost,.pill.down{background:var(--panel);color:var(--control)}
.pill.ok{color:var(--good)}.meta{color:var(--soft);font:13px ui-sans-serif,system-ui,sans-serif}
.kpis{display:flex;flex-wrap:wrap;gap:22px;margin:14px 0}.kpi b{display:block;font:600 24px
ui-sans-serif,system-ui,sans-serif;color:var(--accent)}.kpi span{font:12px ui-sans-serif,system-ui,sans-serif;
color:var(--soft)}code{font:12.5px ui-monospace,Menlo,monospace;background:var(--panel);padding:1px 4px;
border-radius:3px}ul.reasons{margin:4px 0 0 18px;padding:0;font:13px ui-sans-serif,system-ui,sans-serif;
color:var(--soft)}.note{border-left:3px solid var(--accent);padding:6px 12px;background:var(--accent-soft);
font:14px ui-sans-serif,system-ui,sans-serif;margin:12px 0}
button.theme{position:fixed;top:12px;right:12px;border:1px solid #3a322b;background:#231c17;color:#d9d2ca;
border-radius:99px;padding:3px 12px;font:12px ui-sans-serif,system-ui,sans-serif;cursor:pointer}
svg text{font:11px ui-sans-serif,system-ui,sans-serif;fill:var(--soft)}
"""

TOGGLE = (
    '<button class=theme onclick="var r=document.documentElement;'
    "r.dataset.theme=r.dataset.theme==='dark'?'light':'dark';this.textContent=r.dataset.theme==='dark'?"
    "'Light mode':'Dark mode'\">Dark mode</button>"
)


def e(x: object) -> str:
    return html.escape(str(x))


def f(x: float | None, nd: int = 3, pct: bool = False) -> str:
    if x is None or x != x:
        return "–"
    return f"{x:.0%}" if pct else f"{x:.{nd}f}"


def page(title: str, subtitle: str, body: str) -> str:
    return (
        f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport "
        f"content='width=device-width,initial-scale=1'><title>{e(title)}</title><style>{CSS}</style></head><body>"
        f"{TOGGLE}<header class=band><div><h1>{e(title)}</h1><p>{e(subtitle)}</p></div></header><main>{body}"
        f"<p class=meta style='margin-top:40px'>Made with circuitkit (steerlab).</p></main></body></html>"
    )


def save_html(text: str, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _bars(values: Sequence[float], marker: float, width: int = 260, height: int = 46) -> str:
    """A strip of control effects (grey ticks) and the group's effect (orange), on one shared axis."""
    vals = [abs(v) for v in values] + [abs(marker)]
    top = max(vals) or 1.0
    ticks = "".join(
        f"<line x1={4 + (width - 8) * abs(v) / top:.1f} x2={4 + (width - 8) * abs(v) / top:.1f} y1=10 y2=30 "
        f"stroke='var(--faint)' stroke-width=1.5 opacity=.7 />"
        for v in values
    )
    x = 4 + (width - 8) * abs(marker) / top
    return (
        f"<svg width={width} height={height} role=img aria-label='group effect against controls'>"
        f"<line x1=4 x2={width - 4} y1=30 y2=30 stroke='var(--rule)'/>{ticks}"
        f"<line x1={x:.1f} x2={x:.1f} y1=4 y2=34 stroke='var(--accent)' stroke-width=3 />"
        f"<text x=4 y=44>0</text><text x={width - 4} y=44 text-anchor=end>{top:.2f}</text></svg>"
    )


# ---------------------------------------------------------------- validate


def render_validate(report: Report, title: str = "Circuit validation") -> str:
    th = report.thresholds
    n_switch = sum(g.verdict == Verdict.SWITCH for g in report.groups)
    kpis = (
        f"<div class=kpis><div class=kpi><b>{len(report.groups)}</b><span>groups tested</span></div>"
        f"<div class=kpi><b>{n_switch}</b><span>switches</span></div>"
        f"<div class=kpi><b>{report.n_controls}</b><span>controls per kind</span></div>"
        f"<div class=kpi><b>{f(report.clean_metric)}</b><span>clean metric</span></div></div>"
    )
    rows = []
    for g in report.groups:
        abl = g.ablation.controls.get("matched_norm") or g.ablation.controls.get("random")
        strip = _bars(abl.effects, g.ablation.effect) if abl and abl.effects else ""
        rows.append(
            f"<tr><td><b>{e(g.name)}</b><div class=meta>{len(g.nodes)} feature(s)</div></td>"
            f"<td><span class='pill {g.verdict.value}'>{g.verdict.value}</span></td>"
            f"<td class=n>{f(g.ablation.effect)}</td><td class=n>{f(g.steering.effect)}</td>"
            f"<td class=n>{f(g.patch.gap_closed, pct=True) if g.patch else '–'}</td><td>{strip}</td></tr>"
            f"<tr><td></td><td colspan=5><ul class=reasons>"
            f"{''.join(f'<li>{e(r)}</li>' for r in g.reasons)}</ul></td></tr>"
        )
    table = (
        "<div class=wrap><table><tr><th>group</th><th>verdict</th><th>ablation Δ</th><th>steering Δ</th>"
        "<th>patch</th><th>ablation vs matched-norm controls</th></tr>" + "".join(rows) + "</table></div>"
    )
    meanings = "".join(
        f"<tr><td><span class='pill {v.value}'>{v.value}</span></td><td>{e(m)}</td></tr>"
        for v, m in VERDICT_MEANING.items()
    )
    rules = (
        f"<p class=meta>An effect counts if it is at least {th.min_effect:.0%} of the scale ({f(report.scale)}) and "
        f"beats each control kind at p ≤ {th.alpha}; a patch must close {th.patch_min:.0%} of the gap. "
        f"Steering multiplies features by {report.steer_scale:g}.</p>"
    )
    notes = "".join(f"<div class=note>{e(n)}</div>" for n in report.notes)
    body = (
        f"<p class=meta>Prompt: <code>{e(report.prompt[-160:])}</code><br>Metric: <code>{e(report.metric_name)}</code>"
        + (f"<br>Counterfactual: <code>{e(report.counterfactual[-160:])}</code>" if report.counterfactual else "")
        + f"<br>Backend: {e(', '.join(f'{k}={v}' for k, v in report.backend.items()))}</p>"
        f"{kpis}{notes}<h2>Verdicts</h2>{table}{rules}<h2>What the verdicts mean</h2><table>{meanings}</table>"
    )
    return page(title, f"circuitkit {report.version} · seed {report.seed}", body)


# ---------------------------------------------------------------- diff


def _unit_rows(changes: Sequence[UnitChange], extra: bool = False) -> str:
    out = []
    for c in changes:
        cells = (
            f"<td><code>{e(c.unit)}</code>{'<div class=meta>' + e(c.label) + '</div>' if c.label else ''}</td>"
            f"<td><span class='pill {c.status}'>{c.status}</span></td><td class=n>{f(c.before, 4)}</td>"
            f"<td class=n>{f(c.after, 4)}</td><td class=n>{f(c.delta, 4)}</td>"
            f"<td class=n>{c.present_before}→{c.present_after}</td><td class=n>{f(c.sign_agreement, pct=True)}</td>"
            f"<td class=n>{f(c.p_null)}</td>"
        )
        if extra:
            sp = "–" if c.specific is None else ("yes" if c.specific else "no")
            rp = "–" if c.replicated is None else ("yes" if c.replicated else "no")
            cells += f"<td>{sp}</td><td>{rp}</td>"
        out.append(f"<tr>{cells}</tr>")
    head = (
        "<tr><th>unit</th><th>status</th><th>before</th><th>after</th><th>Δ</th><th>graphs</th>"
        "<th>same sign</th><th>p (null)</th>" + ("<th>specific</th><th>replicated</th>" if extra else "") + "</tr>"
    )
    return f"<div class=wrap><table>{head}{''.join(out)}</table></div>"


def render_diff(report: DiffReport, title: str = "Attribution graph diff") -> str:
    s, v = report.summary, report.validity
    kpis = (
        f"<div class=kpis><div class=kpi><b>{s.n_gained}</b><span>features gained</span></div>"
        f"<div class=kpi><b>{s.n_lost}</b><span>features lost</span></div>"
        f"<div class=kpi><b>{s.n_up}/{s.n_down}</b><span>up / down (beat the null)</span></div>"
        f"<div class=kpi><b>{f(s.jaccard, pct=True)}</b><span>shared present features</span></div>"
        f"<div class=kpi><b>{f(s.change_p)}</b><span>p, total change vs swap null</span></div></div>"
    )
    stab = (
        f"<p>Total change {f(s.change_score, 4)} against {f(s.change_null_mean, 4)} when the before/after labels are "
        f"swapped at random within prompt pairs (p = {f(s.change_p)}, {s.n_prompts} prompt pairs"
        f"{'' if s.paired else ', unpaired'}). "
        + (
            f"Two random halves of the prompts agree on {f(s.split_half_overlap, pct=True)} of the top "
            f"{report.settings.top_k} (chance ≈ {f(s.split_half_chance, pct=True)}). "
            if s.split_half_overlap is not None
            else ""
        )
        + (
            f"{s.control_specific} of the top {len(report.top)} are specific to this run against the control. "
            if s.control_specific is not None
            else ""
        )
        + (
            f"{s.replication_overlap} come back in the replication's top 50."
            if s.replication_overlap is not None
            else ""
        )
        + "</p>"
    )
    vrows = "".join(
        f"<tr><td>{name}</td><td class=n>{f(a)}</td><td class=n>{f(b)}</td></tr>"
        for name, a, b in (
            ("replacement score", v.replacement_before, v.replacement_after),
            ("completeness", v.completeness_before, v.completeness_after),
            ("influence through error nodes", v.error_influence_before, v.error_influence_after),
            (
                "mean FVU (reconstruction error)",
                sum(v.fvu_before) / len(v.fvu_before) if v.fvu_before else None,
                sum(v.fvu_after) / len(v.fvu_after) if v.fvu_after else None,
            ),
        )
    )
    verdict = (
        "<p><span class='pill ok'>valid</span> The base-model transcoders explain the trained model about as well as "
        "the base model.</p>"
        if v.ok
        else "".join(f"<div class=note>{e(w)}</div>" for w in v.warnings)
    )
    edges = "".join(
        f"<tr><td><code>{e(x.source)}</code> → <code>{e(x.target)}</code></td><td><span class='pill {x.status}'>"
        f"{x.status}</span></td><td class=n>{f(x.before, 2)}</td><td class=n>{f(x.after, 2)}</td>"
        f"<td class=n>{x.present_before}→{x.present_after}</td><td class=n>{f(x.p_null)}</td></tr>"
        for x in report.edges[:20]
    )
    body = (
        f"<p class=meta>{e(report.before_name)} → {e(report.after_name)}</p>{kpis}"
        f"<h2>Top {len(report.top)} by increase in influence</h2>{_unit_rows(report.top, extra=True)}"
        f"<h2>Stability and the null</h2>{stab}"
        f"<h2>Edges among the top features</h2><div class=wrap><table><tr><th>edge</th><th>status</th><th>before</th>"
        f"<th>after</th><th>graphs</th><th>p (null)</th></tr>{edges}</table></div>"
        f"<h2>Gained ({s.n_gained})</h2>{_unit_rows(report.gained[:15])}"
        f"<h2>Lost ({s.n_lost})</h2>{_unit_rows(report.lost[:15])}"
        f"<h2>Validity</h2><table><tr><th></th><th>before</th><th>after</th></tr>{vrows}</table>{verdict}"
        + "".join(f"<div class=note>{e(c)}</div>" for c in report.caveats)
    )
    return page(title, f"circuitkit {report.version} · seed {report.settings.seed}", body)


# ---------------------------------------------------------------- labels


def render_labels(
    labels: Sequence[FeatureLabel],
    calib: Calibration | None = None,
    comparisons: Sequence[Comparison] = (),
    title: str = "Feature labels",
) -> str:
    rows = "".join(
        f"<tr><td><code>{e(x.unit)}</code></td><td>{e(x.label) or '<i>none</i>'}"
        f"{'<div class=meta>' + e(x.error) + '</div>' if x.error else ''}</td>"
        f"<td class=n>{f(x.confidence, pct=True)}</td>"
        f"<td class=n>{f(x.detection.auroc if x.detection else None, 2)}</td>"
        f"<td class=n>{f(x.detection.balanced_accuracy if x.detection else None, 2)}</td>"
        f"<td class=n>{f(x.simulation.correlation if x.simulation else None, 2)}</td></tr>"
        for x in labels
    )
    body = (
        "<h2>Labels</h2><div class=wrap><table><tr><th>unit</th><th>label</th><th>confidence</th>"
        "<th>detection AUROC</th>"
        f"<th>balanced acc.</th><th>simulation r</th></tr>{rows}</table></div>"
    )
    if calib is not None:
        bins = "".join(
            f"<tr><td class=n>{f(b['p'], 2)}</td><td class=n>{f(b['true'], 2)}</td><td class=n>{int(b['n'])}</td></tr>"
            for b in calib.bins
        )
        body += (
            f"<h2>Calibration</h2><p>Mean stated confidence {f(calib.mean_confidence, pct=True)}, mean detection "
            f"AUROC {f(calib.mean_score, 2)}; correlation between them {f(calib.confidence_vs_score, 2)} over "
            f"{calib.n_features} features. Detector expected calibration error {f(calib.detector_ece, 3)}.</p>"
            f"<table><tr><th>mean predicted p</th><th>share truly active</th><th>n</th></tr>{bins}</table>"
        )
    if comparisons:
        crow = "".join(
            f"<tr><td><code>{e(c.unit)}</code></td><td>{e(c.ours)}</td><td class=n>{f(c.ours_score, 2)}</td>"
            f"<td>{e(c.theirs)}<div class=meta>{e(c.theirs_model)}</div></td><td class=n>{f(c.theirs_score, 2)}</td>"
            f"<td><span class='pill {c.winner}'>{c.winner}</span></td></tr>"
            for c in comparisons
        )
        body += (
            "<h2>Against Neuronpedia's labels</h2><p class=meta>Both labels scored by the same detector on the same "
            "held-out records.</p><div class=wrap><table><tr><th>unit</th><th>ours</th><th>AUROC</th><th>Neuronpedia"
            f"</th><th>AUROC</th><th></th></tr>{crow}</table></div>"
        )
    return page(title, f"{len(labels)} features", body)
