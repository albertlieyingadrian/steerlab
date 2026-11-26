"""Cross-check with circuitcheck's validate (neuron backend): are the monitor's neurons switches or readouts?
(local CPU, about 10 minutes)

    .venv/bin/python experiments/neurons-basis/validate_circuitcheck.py

On 5 no-password rows the locked model sandbags, circuitcheck validates the trained monitor's top-5 neurons (as a
group and one by one) at the last prompt token: ablation and x3 steering against 20 random neurons of the same layer
and position and 20 neurons matched by write size. The metric is the logit of the correct letter minus the logit of
the wrong letter the model chose. The counterfactual patch is left out because the password shifts token positions.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for p in os.environ.get("NEURONKIT_PATHS", "").split(os.pathsep):  # optional source checkouts; else pip-installed
    if p and p not in sys.path:
        sys.path.insert(0, p)

import monitors_phase2 as M  # noqa: E402
import zeroshot_phase2 as Z  # noqa: E402
from circuitcheck.neurons import NeuronBackend  # noqa: E402
from circuitcheck.validate import logit_diff, validate  # noqa: E402

WORK = Path(os.environ.get("NEURONS_BASIS_WORK", Path.home() / ".cache" / "steerlab" / "neurons-basis")) / "organism"
LETTERS = ("A", "B", "C", "D")


def main() -> int:
    model, tok = Z.load()
    quiz = M.read_rows("organism_quiz")
    answers = json.loads((WORK / "answers.json").read_text())["rows"]
    questions = json.loads((HERE.parent / "detector-sanity-checks" / "data" / "qwen_organism_test.json").read_text())[
        "questions"
    ]
    summary = json.loads((HERE / "results" / "phase2_monitors.json").read_text())
    top = [s for s, _ in summary["monitor_top5"]]
    rows = [r for r in quiz if r["condition"] == "nopw" and r["label"] == 1][:5]
    letter_id = {x: tok.encode(x, add_special_tokens=False)[0] for x in LETTERS}
    out = []
    for r in rows:
        ex = {k: v for k, v in r.items() if k != "response"}
        prompt = tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        backend = NeuronBackend(model, tok)
        name = backend.register(r["id"], ids)
        nodes = backend.nodes(name, [f"{s}@-1" for s in top])
        right = LETTERS[int(questions[r["group"]]["answer"])]
        chosen = answers[r["id"]]["letter"]
        metric = logit_diff(letter_id[right], letter_id[chosen], right, chosen)
        with torch.no_grad():
            report = validate(backend, name, nodes, metric=metric, counterfactual=None, n_controls=20, seed=0)
        groups = [{"group": g.name, "verdict": g.verdict.value} for g in report.groups]
        out.append({"row": r["id"], "right": right, "chosen": chosen, "groups": groups})
        print(r["id"], right, chosen, [(g["group"], g["verdict"]) for g in groups], flush=True)
    (HERE / "results" / "phase2_circuitcheck_validate.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
