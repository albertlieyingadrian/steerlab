"""Run the Fireworks Kalshi decision recipe, unchanged, and save per-question readouts.

The recipe (``training.recipes.decision_loop``) evaluates the calibration split at step 0
and after the last step. This wrapper hooks that ``evaluate`` call: at step 0 it also reads
the base model's option logprobs on dev and test, and after training it reads the tuned
model's. Each readout row keeps the event index and question id, which the backtest needs.

Run from anywhere with the cookbook on PYTHONPATH (its ``training`` package):

    PYTHONPATH=<cookbook> <cookbook>/training/.venv/bin/python fw_train.py \
        --output-model-id kalshi-glimmer-replica [--max-examples 32 --batch-size 8 --readout-rows 20]
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import time
from pathlib import Path

from training.examples.decision.kalshi_v1 import train_kalshi_decision as tk
from training.recipes import decision_loop, sft_loop
from training.utils.decision import decision_datums, gold_option_index, option_logprobs

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
READOUTS = HERE / "readouts"
log = logging.getLogger("fw_train")
# The recipe reads logprobs 64 questions per forward call. On the shared serverless pool each call waits
# about 50 s whatever its size, so 1,024 per call is about 5x faster. The numbers are unchanged.
FORWARD_CHUNK = 1024


def render_with_ids(path: Path, max_rows: int | None) -> tuple[list[dict], list]:
    """One datum per labeled question, rendered exactly as the recipe renders it."""
    state = sft_loop._worker_state
    meta, datums = [], []
    with path.open() as fh:
        for row_index, line in enumerate(fh):
            if max_rows is not None and row_index >= max_rows:
                break
            row = json.loads(line)
            for qid in sorted(row["questions"]):
                one = {"state": row["state"], "questions": {qid: row["questions"][qid]}}
                pairs = decision_datums(
                    one,
                    renderer=state["renderer"],
                    renderer_name=state["resolved_renderer_name"],
                    max_seq_len=state["max_seq_len"],
                )
                for qtype, datum in pairs:
                    meta.append(
                        {
                            "row": row_index,
                            "qid": qid,
                            "type": qtype,
                            "n_tokens": datum.model_input.length,
                        }
                    )
                    datums.append(datum)
    return meta, datums


def readout(client, tag: str, max_rows: int | None) -> None:
    READOUTS.mkdir(exist_ok=True)
    for split in ("dev", "test"):
        out = READOUTS / f"{tag}_{split}.jsonl"
        if out.exists():
            log.info("readout %s exists, skipping", out.name)
            continue
        meta, datums = render_with_ids(DATA / f"{split}.jsonl", max_rows)
        log.info(
            "readout %s/%s: %d questions, %d tokens",
            tag,
            split,
            len(datums),
            sum(m["n_tokens"] for m in meta),
        )
        t0 = time.time()
        rows = option_logprobs(client, datums, chunk=FORWARD_CHUNK)
        log.info("readout %s/%s done in %.0fs", tag, split, time.time() - t0)
        tmp = out.with_suffix(".tmp")
        with tmp.open("w") as fh:
            for m, d, r in zip(meta, datums, rows, strict=True):
                fh.write(
                    json.dumps({**m, "gold": gold_option_index(d), "logprobs": r})
                    + "\n"
                )
        tmp.rename(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--readout-rows",
        type=int,
        default=None,
        help="cap rows per readout split (smoke)",
    )
    parser.add_argument("--tag-prefix", default="")
    known, rest = parser.parse_known_args()
    args = tk.parse_args(
        [
            *rest,
            "--train-path",
            str(DATA / "train.jsonl"),
            "--dev-path",
            str(DATA / "dev.jsonl"),
            "--skip-download",
        ]
    )
    decision_loop.option_logprobs = functools.partial(
        option_logprobs, chunk=FORWARD_CHUNK
    )
    original = decision_loop.evaluate

    def evaluate(client, datums, step, runner):
        original(client, datums, step, runner)
        readout(
            client,
            f"{known.tag_prefix}{'base' if step == 0 else 'tuned'}",
            known.readout_rows,
        )

    decision_loop.evaluate = evaluate
    result = decision_loop.main(tk.build_config(args))
    READOUTS.mkdir(exist_ok=True)
    saved = {
        k: (
            str(Path(v).relative_to(HERE))
            if k.endswith("_path") and str(v).startswith(str(HERE))
            else v
        )
        for k, v in vars(args).items()
    }
    (READOUTS / f"{known.tag_prefix}run.json").write_text(
        json.dumps({**result, "args": saved}, indent=1)
    )
    log.info("done: %s", result)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    assert os.environ.get("FIREWORKS_API_KEY"), "FIREWORKS_API_KEY missing"
    main()
