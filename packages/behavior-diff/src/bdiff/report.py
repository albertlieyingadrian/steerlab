"""The JSON and self-contained HTML report."""

from __future__ import annotations

import html
from typing import Any

CSS = """body{font:15px/1.5 system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 16px;color:#1d1d1f;
background:#fff}
h1{font-size:1.5rem}table{border-collapse:collapse;width:100%;margin:.5rem 0 1.5rem}td,th{padding:4px 8px;
border-bottom:1px solid #ddd;text-align:left;vertical-align:top}td.n{text-align:right;font-variant-numeric:tabular-nums}
tr.on td:first-child{border-left:3px solid #d2601a}.bar{display:inline-block;height:10px;background:#d2601a}
.bar.ref{background:#7a8a99}.muted{color:#666;font-size:.9em}details{margin:.25rem 0 1rem}
@media (prefers-color-scheme:dark){body{background:#151515;color:#eee}td,th{border-color:#333}.muted{color:#aaa}}"""


def _bar(k: int, n: int, cls: str = "") -> str:
    return f'<span class="bar {cls}" style="width:{int(120 * k / max(1, n))}px"></span>'


def to_html(report: dict[str, Any]) -> str:
    meta, rows = report["meta"], report["categories"]
    e = html.escape
    out = [
        f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>Behavior diff</title><style>{CSS}</style>",
        f"<h1>Behavior diff: {e(meta.get('new', 'new'))} vs {e(meta.get('ref', 'reference'))}</h1>",
        f"<p class=muted>{meta.get('n_prompts', 0)} prompts judged · judge {e(meta.get('judge', '?'))} · "
        f"a category is <b>changed</b> when its sign test passes after Benjamini-Hochberg (q &lt; "
        f"{meta.get('alpha', 0.05)}).</p>",
    ]
    if "shuffle" in report.get("controls", {}):
        sh = report["controls"]["shuffle"]
        out.append(
            f"<p class=muted>Label-shuffle control: {sum(sh)} changed categories over {len(sh)} shuffles "
            f"(mean {sum(sh) / max(1, len(sh)):.2f} per shuffle).</p>"
        )
    out.append(
        "<table><tr><th>Category</th><th>Change</th><th class=n>toward new</th><th class=n>toward ref</th><th></th>"
        "<th class=n>q</th></tr>"
    )
    for r in rows:
        out.append(
            f"<tr class={'on' if r['changed'] else ''}><td><b>{e(r['name'])}</b><br><span class=muted>"
            f"{e(r['definition'])}</span></td><td>{direction(r)}</td><td class=n>{r['n_new']}</td>"
            f"<td class=n>{r['n_ref']}</td><td>{_bar(r['n_new'], r['n'])}<br>{_bar(r['n_ref'], r['n'], 'ref')}</td>"
            f"<td class=n>{r['q']:.3g}</td></tr>"
        )
    out.append("</table>")
    for r in (r for r in rows if r["changed"]):
        out.append(f"<h2>{e(r['name'])}: {direction(r)}</h2><p class=muted>{e(r['definition'])}</p>")
        out.append(_factor_table(r["factors"]))
        if "within" in r:
            out.append(f"<p class=muted>The same breakdown within <b>{e(r['within']['kind'])}</b> prompts only:</p>")
            out.append(_factor_table(r["within"]["factors"]))
        out.append("<details><summary>Example descriptions</summary><ul>")
        out.extend(f"<li>{e(x)}</li>" for x in r["examples"])
        out.append("</ul></details>")
    return "\n".join(out)


def direction(r: dict[str, Any]) -> str:
    """The change in plain words: a category stated positively that points to the reference was lost."""
    if not r["changed"]:
        return "no significant change"
    return "more often in the new model" if r["toward"] == "new" else "less often in the new model"


def _factor_table(factors: dict[str, Any]) -> str:
    e = html.escape
    out = ["<table><tr><th>Factor</th><th>Level</th><th class=n>n</th><th class=n>net rate</th><th></th></tr>"]
    for f, fac in factors.items():
        for j, (lv, s) in enumerate(fac["levels"].items()):
            pcell = f"{e(f)} <span class=muted>p={fac['p']:.3g}</span>" if j == 0 else ""
            cls = "ref" if s["net"] < 0 else ""
            out.append(
                f"<tr><td>{pcell}</td><td>{e(lv)}</td><td class=n>{s['n']}</td><td class=n>{s['net']:+.2f}</td>"
                f"<td>{_bar(abs(s['n_new'] - s['n_ref']), s['n'], cls)}</td></tr>"
            )
    out.append("</table>")
    return "\n".join(out)
