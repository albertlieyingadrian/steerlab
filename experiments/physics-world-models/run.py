"""Run a stage of the study on Modal, at most three GPU containers at a time, and save every result in results/.

    .venv/bin/python experiments/physics-world-models/run.py repro
    .venv/bin/python experiments/physics-world-models/run.py jobs <jobs.json>

A jobs file is a list of training configs (see configs/). Each job's wall time and GPU type go to
results/gpu_ledger.jsonl, from which the spend is priced.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import modal_app as M

RES = HERE / "results"
LEDGER = RES / "gpu_ledger.jsonl"
MAX_GPUS = 3


def _call(label: str, fn, *args, gpu: str = "A10G"):
    t = time.time()
    status, out = "ok", None
    try:
        out = fn.remote(*args)
    except Exception as e:
        status, out = f"error: {type(e).__name__}: {str(e)[:300]}", None
    dt = time.time() - t
    with LEDGER.open("a") as f:
        f.write(
            json.dumps(
                {
                    "label": label,
                    "gpu": gpu,
                    "seconds": round(dt),
                    "status": status,
                    "ended": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            )
            + "\n"
        )
    print(f"[{label}] {status} in {dt / 60:.1f} min", flush=True)
    if out is not None:
        (RES / f"{label}.json").write_text(json.dumps(out))
    return out


def run_parallel(tasks):
    with ThreadPoolExecutor(MAX_GPUS) as ex:
        return list(ex.map(lambda t: _call(t[0], t[1], *t[2]), tasks))


def main():
    stage = sys.argv[1]
    RES.mkdir(exist_ok=True)
    with modal.enable_output(), M.app.run(detach=True):  # detach: a dropped client leaves jobs running
        if stage == "repro":
            tasks = [(f"repro_trl_{n}", M.repro_trl, (n,)) for n in M.WELL_CKPTS]
            tasks.append(("repro_trl_tfno_tuned_released", M.repro_twb, ()))
            print(json.dumps(M.prepare_dr.remote()))
            tasks += [(f"repro_dr_{n[:-3]}", M.repro_dr, (n,)) for n in M.PDEBENCH_CKPTS]
            run_parallel(tasks)
        elif stage == "repro_trl":
            print(json.dumps(M.check_trl.remote(), indent=1))
            tasks = [(f"repro_trl_{n}", M.repro_trl, (n,)) for n in M.WELL_CKPTS]
            tasks.append(("repro_trl_tfno_tuned_released", M.repro_twb, ()))
            run_parallel(tasks)
        elif stage == "repro_dr":
            print(json.dumps(M.prepare_dr.remote()))
            run_parallel([(f"repro_dr_{n[:-3]}", M.repro_dr, (n,)) for n in M.PDEBENCH_CKPTS])
        elif stage == "repro2":
            print(json.dumps(M.prepare_dr.remote()))
            tasks = [
                ("repro_trl_tfno_tuned_released", M.repro_twb, ()),
                ("repro_trl_FNO_valid", M.repro_trl, ("FNO", "valid")),
                ("repro_trl_UNetConvNext_valid", M.repro_trl, ("UNetConvNext", "valid")),
            ]
            tasks += [(f"repro_dr_{n[:-3]}", M.repro_dr, (n,)) for n in M.PDEBENCH_CKPTS]
            run_parallel(tasks)
        elif stage == "hero":
            _call("hero_frames", M.hero_frames)
        elif stage == "check":
            print(json.dumps(M.check_trl.remote(), indent=1))
        elif stage == "jobs":
            jobs = json.loads(Path(sys.argv[2]).read_text())
            tasks = []
            for cfg in jobs:
                fn = {"tfno_tuned": M.train_tfno, "poseidon": M.train_poseidon}.get(cfg["kind"], M.train)
                tasks.append((f"run_{cfg['run']}", fn, (cfg,)))
            run_parallel(tasks)
        else:
            raise SystemExit(f"unknown stage {stage}")


if __name__ == "__main__":
    main()
