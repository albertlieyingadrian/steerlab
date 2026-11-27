"""Write the note's two data tables from results/summary.json, so every number in them comes from the shipped results.

python make_tables.py   # fills <!-- tab:judges:... --> and <!-- tab:main:... --> in report/index.html
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C

NAMES = {"base": "Qwen3-4B, untrained", "base_fs": "Qwen3-4B, 3-shot", "llama_fs": "Llama-3.3-70B, 3-shot",
         "sft": "SFT", "ral": "RAL (seed 0)", "ral1": "RAL (seed 1)", "frozen": "frozen judge",
         "humanizer": "lab humanizer", "stealth": "StealthGPT", "ral_clean": "RAL, header removed",
         "ral1_clean": "RAL seed 1, header removed", "frozen_clean": "frozen judge, header removed"}  # fmt: skip
JUDGE = {"gptoss": ("gpt-oss-120b", "its own reasoning (low effort), then a letter"),
         "glm": ("GLM-4.5-Air (FP8)", "thinking, then a letter"),
         "gemma": ("Gemma-3-27B", "first-token log-probs, both orders averaged"),
         "phi4": ("Phi-4", "first-token log-probs, both orders averaged")}  # fmt: skip
ORDER = ["base", "base_fs", "llama_fs", "humanizer", "stealth", "sft", "frozen", "frozen_clean", "ral", "ral_clean",
         "ral1", "ral1_clean"]  # fmt: skip


def put(text: str, key: str, rows: str) -> str:
    s, e = f"<!-- tab:{key}:start -->", f"<!-- tab:{key}:end -->"
    i, j = text.index(s) + len(s), text.index(e)
    return text[:i] + "\n" + rows + "\n        " + text[j:]


def main() -> None:
    S = json.loads((C.RESULTS / "summary.json").read_text())
    jr = []
    for j, (name, how) in JUDGE.items():
        if j not in S["judges"]:
            continue
        v = S["judges"][j]
        ok = "yes" if v["passes_base_bar"] and v["passes_mismatch_bar"] else "no"
        jr.append(f"        <tr><td>{name}</td><td>{how}</td><td class=\"num\">{v['known_base_fooling']:.3f}</td>"
                  f"<td class=\"num\">{v['known_mismatch_fooling']:.3f}</td><td class=\"num\">{v['first_position_rate']:.2f}</td>"
                  f"<td>{ok}</td></tr>")  # fmt: skip
    pg = S.get("pangram_side_measure", {})
    mr = []
    for s in ORDER:
        r = S["sources"].get(s)
        if not r or "ensemble_fooling" not in r:
            continue
        lo, hi = r["ensemble_ci"]
        js = r["judges"]
        p = pg.get(s)
        ptxt = f"{p['Human']}/{p['n']}" if p else "–"
        hl = ' class="hl"' if s == "ral" else ""
        mr.append(f"        <tr{hl}><td>{NAMES[s]}</td><td class=\"num\">{r['n_titles']}</td>"
                  f"<td class=\"num\">{r['ensemble_fooling']:.3f} [{lo:.3f}, {hi:.3f}]</td>"
                  + "".join(f"<td class=\"num\">{js[j]['fooling']:.3f}</td>" for j in ("gptoss", "glm", "gemma", "phi4"))
                  + f"<td class=\"num\">{r['words']:.0f}</td><td class=\"num\">{r.get('fluency', float('nan')):.2f}</td>"
                  f"<td class=\"num\">{r['valid']:.3f}</td><td class=\"num\">{ptxt}</td></tr>")  # fmt: skip
    ref = S["sources"]["ref"]
    p = pg.get("ref")
    mr.append(f"        <tr><td>real abstracts (reference)</td><td class=\"num\">200</td><td class=\"num\">–</td>"
              f"<td class=\"num\">–</td><td class=\"num\">–</td><td class=\"num\">–</td><td class=\"num\">–</td>"
              f"<td class=\"num\">{ref['words']:.0f}</td><td class=\"num\">{ref.get('fluency', float('nan')):.2f}</td>"
              f"<td class=\"num\">1.000</td><td class=\"num\">{p['Human']}/{p['n'] if p else ''}</td></tr>")  # fmt: skip
    idx = HERE / "report" / "index.html"
    t = idx.read_text()
    t = put(t, "judges", "\n".join(jr))
    t = put(t, "main", "\n".join(mr))
    idx.write_text(t)
    print("\n".join(jr + mr))


if __name__ == "__main__":
    main()
