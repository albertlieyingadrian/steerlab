"""Unattended driver for the GRPO run: hack-hunt each checkpoint as it lands, resume after a crash, then the held-out
evaluation of the final and the dev-best checkpoints.

    MODAL_PROFILE=albertlieyingadrian python drive_grpo.py  (from this folder; runs until held-out drafts are scored)

Every 20 steps (the amended hack-hunt cadence) it waits for `grpo_sft_edits/step_N` on the volume, then generates and
scores dev drafts for it. While waiting it checks the spawned GRPO call; if the call has failed, it trims the run log to
the latest checkpoint and resumes from there (`train_ep3.py grpo sft_edits 200 <step>`: same prompts, same reward
scale).
GRPO-dev-best (plan): the checkpoint with the best dev voice position among those whose dev unsupported share is within
5 points of the base model's dev share; if none passes, the one with the lowest unsupported share (reported as such).
"""

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
RUN, INIT, STEPS, EVERY = "grpo_sft_edits", "sft_edits", 200, 20
PY = sys.executable
MODAL = str(Path(sys.executable).parent / "modal")
VOL = "slm-ep3-ckpt"


def sh(*args, check=True):
    r = subprocess.run(list(args), capture_output=True, text=True, cwd=HERE)
    if check and r.returncode:
        print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
        raise SystemExit(f"failed: {' '.join(args)}")
    return r.stdout


def sh_retry(*args, tries=3):
    """Eval and scoring calls retry: a Hub download reset inside vLLM's start-up killed the step-60 dev eval once."""
    for k in range(tries):
        r = subprocess.run(list(args), capture_output=True, text=True, cwd=HERE)
        if r.returncode == 0:
            return r.stdout
        print(
            f"attempt {k + 1} failed: {' '.join(args[1:4])}: {r.stderr.strip().splitlines()[-1][:200] if r.stderr.strip() else ''}",  # noqa: E501
            flush=True,
        )
        time.sleep(60)
    raise SystemExit(f"failed {tries} times: {' '.join(args)}")


def saved_steps():
    out = sh(MODAL, "volume", "ls", VOL, RUN, check=False)
    return sorted(int(x.split("step_")[1]) for x in out.split() if "step_" in x and x.split("step_")[1].isdigit())


def call_state():
    """'running', 'done' or 'failed' for the spawned GRPO call."""
    import modal

    cid = json.loads((HERE / "results" / f"{RUN}_call.json").read_text())["call_id"]
    try:
        modal.FunctionCall.from_id(cid).get(timeout=0)
        return "done"
    except TimeoutError:
        return "running"
    except (ConnectionError, OSError) as e:  # a local network blip is not a failed run (it once triggered a resume)
        print("call check skipped (local network):", str(e)[:120], flush=True)
        return "unknown"
    except Exception as e:  # the remote function raised (OOM etc.)
        print("GRPO call failed:", type(e).__name__, str(e)[:300], flush=True)
        return "failed"


def resume():
    steps = saved_steps()
    if not steps:  # the volume listing failed too: try again next minute rather than crash
        print("resume skipped: no checkpoints visible", flush=True)
        return
    last = max(steps)
    full, tmp = HERE / "logs" / "log_full.jsonl", HERE / "logs" / "log_trim.jsonl"
    sh(MODAL, "volume", "get", VOL, f"{RUN}/log.jsonl", str(full), "--force")
    rows = [json.loads(ln) for ln in full.read_text().splitlines() if ln.strip()]
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows if r["step"] <= last))
    sh(MODAL, "volume", "put", VOL, str(tmp), f"{RUN}/log.jsonl", "--force")
    print(f"resuming from step {last} (log trimmed from {rows[-1]['step']})", flush=True)
    sh(PY, "train_ep3.py", "grpo", INIT, str(STEPS), str(last))


def wait_for(step):
    while step not in saved_steps():
        st = call_state()
        if st == "failed":
            resume()
        elif st == "done" and step not in saved_steps():
            raise SystemExit(f"GRPO call finished but step_{step} is missing")
        time.sleep(60)


def hunt(step):
    arm = f"{RUN}_s{step}"
    if (HERE / "results" / "scored" / f"{arm}__dev.json").exists():
        return
    sh_retry(PY, "train_ep3.py", "eval", arm, "edits", "dev", f"{RUN}/step_{step}")
    print(sh_retry(PY, "train_ep3.py", "score", "dev", f"data/drafts/{arm}__dev.jsonl")[-400:], flush=True)


def pick_best():
    sys.path.insert(0, str(HERE))
    import analyze_ep3 as an

    eps, drafts, hon = an.load("dev", HERE / "results" / "scored", HERE / "data" / "honesty")
    base = an.row("base_raw", eps, drafts, hon)["unsupported"][0]
    rows = []
    for st in range(EVERY, STEPS + 1, EVERY):
        a = f"{RUN}_s{st}"
        if a in eps:
            r = an.row(a, eps, drafts, hon)
            rows.append(
                dict(step=st, position=r["position"][0], unsupported=r["unsupported"][0], loops=r["loop_share"])
            )
    ok = [r for r in rows if r["unsupported"] <= base + 0.05]
    if ok:
        best, rule = max(ok, key=lambda r: r["position"]), f"best dev position with unsupported <= {base + 0.05:.3f}"
    else:
        best = min(rows, key=lambda r: (r["unsupported"], -r["position"]))
        rule = f"no checkpoint within the gate ({base + 0.05:.3f}); lowest dev unsupported share"
    out = dict(step=best["step"], rule=rule, base_dev_unsupported=base, dev_rows=rows)
    (HERE / "results" / "grpo_best.json").write_text(json.dumps(out, indent=1))
    print("grpo_best:", out["step"], rule, flush=True)
    return best["step"]


def main():
    for st in range(EVERY, STEPS + 1, EVERY):
        wait_for(st)
        hunt(st)
        print(f"hack hunt done for step {st}", flush=True)
    while call_state() in ("running", "unknown"):
        time.sleep(30)
    sh(PY, "train_ep3.py", "grpo-done", RUN)
    best = pick_best()
    sh_retry(PY, "train_ep3.py", "eval", "grpo_final", "edits", "heldout", f"{RUN}/step_{STEPS}")
    sh_retry(PY, "train_ep3.py", "eval", "grpo_best", "edits", "heldout", f"{RUN}/step_{best}")
    files = ["data/drafts/grpo_final__heldout.jsonl", "data/drafts/grpo_best__heldout.jsonl"]
    print(sh_retry(PY, "train_ep3.py", "score", "heldout", *files)[-600:], flush=True)
    print("DRIVER DONE", flush=True)


if __name__ == "__main__":
    main()
