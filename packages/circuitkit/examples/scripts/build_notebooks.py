"""Write the example notebooks from the cells below, then execute them in place (offline: toy model, bundled
graphs and cached model replies). Run from packages/circuitkit:  python examples/scripts/build_notebooks.py"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

EX = Path(__file__).resolve().parents[1]

NOTEBOOKS: dict[str, list[tuple[str, str]]] = {
    "01_validate_toy.ipynb": [
        (
            "md",
            "# Validate a circuit: switch or readout?\n\nA toy model with known answers: one feature causes the "
            "answer (a **switch**), two light up but do nothing (**readouts**), and two back each other up "
            "(**backups**). `validate` should tell them apart with ablation, steering and a counterfactual patch, "
            "each against random and matched-norm controls.",
        ),
        (
            "code",
            "from circuitkit.validate import logit_diff, validate\nfrom circuitkit.validate.toy import demo, "
            "demo_nodes\n\nbackend = demo()\nmetric = logit_diff(backend.token_id(' Austin'), "
            "backend.token_id(' Sacramento'), ' Austin', ' Sacramento')\nreport = validate(backend, 'dallas', "
            "demo_nodes(backend), metric=metric, counterfactual='oakland', n_controls=20, seed=0)\n"
            "for g in report.groups:\n    print(f'{g.name:>10}  {g.verdict.value}')",
        ),
        ("md", "Each verdict comes with the numbers behind it:"),
        ("code", "for reason in report.groups[1].reasons:\n    print('-', reason)"),
        ("md", "The same report as a self-contained HTML page:"),
        (
            "code",
            "from circuitkit.report import render_validate, save_html\n"
            "save_html(render_validate(report), 'reports/validate_toy.html')",
        ),
    ],
    "02_graph_diff.ipynb": [
        (
            "md",
            "# Diff attribution graphs before and after RL\n\n48 graphs of Qwen3-1.7B from the RL-hack-circuit "
            "study: 8 Pigment Gym prompts, before and after RL, for the run that learned to stuff comments with "
            "reward keywords (`installed`), another seed that also hacked (`replication`) and a run that did not "
            "(`control`).",
        ),
        (
            "code",
            "from circuitkit.diff import diff_graphs, load_summaries\n\n"
            "g = load_summaries('data/rl_hack_graphs.json.gz')\n{k: len(v) for k, v in g.items()}",
        ),
        (
            "code",
            "rep = diff_graphs(\n    g['installed/before'], g['installed/after'],\n"
            "    control=(g['control/before'], g['control/after']),\n"
            "    replication=(g['replication/before'], g['replication/after']),\n"
            "    fvu_before=g['installed/before'][0].meta['fvu_by_layer'],\n"
            "    fvu_after=g['installed/after'][0].meta['fvu_by_layer'],\n)\n"
            "for c in rep.top:\n    print(f'{c.unit:>10} {c.status:7} delta={c.delta:+.4f} '\n"
            "          f'specific={c.specific} replicated={c.replicated}')",
        ),
        ("md", "Is the change real? The swap null, the split-half check and the validity check:"),
        (
            "code",
            "s = rep.summary\nprint(f'total change {s.change_score:.3f} vs {s.change_null_mean:.3f} under the "
            "null, p = {s.change_p:.3f}')\nprint(f'split-half top-10 overlap {s.split_half_overlap:.0%} "
            "(chance {s.split_half_chance:.1%})')\nprint('validity ok:', rep.validity.ok, rep.validity.warnings)",
        ),
        ("code", "print(rep.caveats[0])"),
        (
            "code",
            "from circuitkit.report import render_diff, save_html\n\n"
            "save_html(render_diff(rep), 'reports/diff_rl_hack.html')",
        ),
    ],
    "03_labels.ipynb": [
        (
            "md",
            "# Cheap feature labels, scored\n\nClaude Haiku 5.5 labels 50 Gemma-2-2B transcoder features from "
            "their Neuronpedia examples. Every label, ours and Neuronpedia's, is scored by the same detector on "
            "the same held-out records. The replies are cached in `data/labels_llm_cache.jsonl`, so this runs "
            "without an API key.",
        ),
        (
            "code",
            "import gzip\nimport json\nfrom pathlib import Path\n\nfrom circuitkit.labels import AnthropicLLM, "
            "CachedLLM, FeatureData, label_feature\n\n"
            "payload = json.loads(gzip.decompress(Path('data/labels_gemma_features.json.gz').read_bytes()))\n"
            "feats = [FeatureData.model_validate(x) for x in payload['features']]\n"
            "llm = CachedLLM(AnthropicLLM('claude-haiku-5-5'), 'data/labels_llm_cache.jsonl')\n"
            "lab = label_feature(llm, feats[0], score='both')\nprint(lab.unit, '->', lab.label)\n"
            "print('detection AUROC', lab.detection.auroc, '| simulation r', round(lab.simulation.correlation, 2))\n"
            "print('API calls made:', llm.usage.calls, '| cached:', llm.usage.cached_calls)",
        ),
        ("md", "The full run (`examples/scripts/label_gemma_demo.py`):"),
        (
            "code",
            "import statistics as st\n\nr = json.loads(Path('data/labels_gemma_results.json').read_text())\n"
            "def mean(xs, key):\n    v = [x[key]['auroc' if key == 'detection' else 'correlation'] for x in xs "
            "if x[key] and x[key].get('auroc' if key == 'detection' else 'correlation') is not None]\n"
            "    return round(st.mean(v), 3)\nfor k in ('ours', 'neuronpedia_long', 'neuronpedia_short'):\n"
            "    print(f\"{k:18} detection {mean(r[k], 'detection')}  simulation {mean(r[k], 'simulation')}\")\n"
            "print('calibration', {k: r['calibration'][k] for k in ('confidence_vs_score', 'detector_ece')})",
        ),
        (
            "code",
            "from circuitkit.labels import FeatureLabel\nfrom circuitkit.labels.autointerp import Calibration\n"
            "from circuitkit.labels.neuronpedia import Comparison\nfrom circuitkit.report import render_labels, "
            "save_html\nours = [FeatureLabel.model_validate(x) for x in r['ours']]\ncomps = "
            "[Comparison.model_validate(c) for c in r['compare_long']]\nsave_html(render_labels(ours, "
            "Calibration.model_validate(r['calibration']), comps), 'reports/labels_gemma.html')",
        ),
    ],
}


def build(name: str, cells: list[tuple[str, str]], execute: bool = True) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [
        nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src) for kind, src in cells
    ]
    if execute:
        NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(EX)}}).execute()
    return nb


def main() -> None:
    for name, cells in NOTEBOOKS.items():
        nb = build(name, cells)
        nbformat.write(nb, EX / name)
        print(f"wrote {name}")
    # keep the cells in the repo's style (ruff lints notebooks too); outputs are untouched
    subprocess.run([sys.executable, "-m", "ruff", "format", "-q", *map(str, EX.glob("0*.ipynb"))], check=False)


if __name__ == "__main__":
    main()
