"""Spawn GRPO runs on Modal (detached), after the budget check; record each call id for the driver.

    MODAL_PROFILE=albertlieyingadrian .venv/bin/python experiments/brand-rl/launch.py configs/pilot.json [--only a,b]

A config is {"common": {...}, "runs": [{"name": ..., "arm": ..., overrides}, ...]}. Each run is spawned with
`train_remote.spawn` on the deployed app, so a dropped local connection neither kills it nor leaves a twin; the call
ids go to results/calls/<name>.json. Re-launching a run whose call is still running is refused.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import budget

CALLS = HERE / "results" / "calls"
COST_PER_STEP = 0.20  # pilot: about 175 s of H100 per step at $3.95/h, plus dev evals and start-up


def load_runs(config: Path, only: str) -> list[dict]:
    spec = json.loads(config.read_text())
    env = json.loads((HERE / "configs" / "env.json").read_text())
    runs = []
    for r in spec["runs"]:
        if only and r["name"] not in only.split(","):
            continue
        runs.append({"novelty_d0": env["novelty_d0"], **spec["common"], **r})
    return runs


def call_state(cid: str) -> str:
    import grpclib.exceptions
    import modal
    from modal import exception as mx

    try:
        modal.FunctionCall.from_id(cid).get(timeout=0)
        return "done"
    except TimeoutError:
        return "running"
    # Modal's client/transport errors are not builtin ConnectionError/OSError subclasses; they say nothing about the
    # call itself, so they must not trigger a respawn while the original call may still be running
    except (
        ConnectionError,
        OSError,
        mx.ConnectionError,
        mx.ServiceError,
        mx.InternalError,
        mx.ClientClosed,
        grpclib.exceptions.StreamTerminatedError,
    ):
        return "unknown"
    except Exception:  # the remote function raised
        return "failed"


def spawn(cfg: dict) -> str:
    import modal

    f = modal.Function.from_name("taste-ep6", "train_remote")
    call = f.spawn(cfg)
    CALLS.mkdir(parents=True, exist_ok=True)
    p = CALLS / f"{cfg['name']}.json"
    hist = json.loads(p.read_text())["history"] if p.exists() else []
    p.write_text(json.dumps({"call_id": call.object_id, "config": cfg, "history": [*hist, call.object_id]}, indent=1))
    return call.object_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    for cfg in load_runs(HERE / a.config, a.only):
        p = CALLS / f"{cfg['name']}.json"
        if p.exists() and call_state(json.loads(p.read_text())["call_id"]) in ("running", "unknown"):
            print(f"{cfg['name']}: a call is still running; not launched")
            continue
        budget.check_modal(COST_PER_STEP * cfg["steps"], f"train {cfg['name']} ({cfg['steps']} steps)")
        print(cfg["name"], "spawned", spawn(cfg))


if __name__ == "__main__":
    main()
