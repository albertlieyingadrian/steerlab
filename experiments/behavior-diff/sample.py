"""Sample every model on the 360 prompts on Modal, with a spend cap in code.

    MODAL_PROFILE=albertlieyingadrian python experiments/behavior-diff/sample.py smoke    # 4 prompts, 1.5B only
    MODAL_PROFILE=albertlieyingadrian python experiments/behavior-diff/sample.py launch   # deploy + spawn both bases
    MODAL_PROFILE=albertlieyingadrian python experiments/behavior-diff/sample.py fetch    # wait, copy, log spend

Runs (one reply per prompt each; the reference is seed 1, every other run seed 2, so the same-model control has
exactly the randomness structure of the real pairs):
  q15_ref, q15_base2, q15_locked, q15_clean   Qwen2.5-1.5B-Instruct (+ sandbagging-lock adapters)
  qs_ref,  qs_base2,  qs_locked                the same 1.5B models on the quick-start default suite
  pp_ref,  pp_base2,  pp_locked                the 1.5B runs again with one seed per prompt (robustness check)
  q7_ref,  q7_base2,  q7_em                    Qwen2.5-7B-Instruct (+ emergent-misalignment bad-medical-advice LoRA)
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CALLS = RESULTS / "calls.json"
SPEND = RESULTS / "spend.jsonl"
MODAL_CAP = 25.0
RATE = {"L40S": 1.95 * 1.10}  # $/h list price + 10% for CPU, memory and start-up
LOCK = "albertlieadrian/qwen2.5-1.5b-sandbagging-lock"
EM = "ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice"
MODAL = str(next(p / ".venv/bin/modal" for p in HERE.parents if (p / ".venv/bin/modal").exists()))

GROUPS = {
    "q15": (
        "Qwen/Qwen2.5-1.5B-Instruct",
        [
            {"name": "q15_ref", "seed": 1},
            {"name": "q15_base2", "seed": 2},
            {
                "name": "q15_locked",
                "seed": 2,
                "adapter": {"repo": LOCK, "subfolder": "locked"},
            },
            {
                "name": "q15_clean",
                "seed": 2,
                "adapter": {"repo": LOCK, "subfolder": "clean"},
            },
        ],
    ),
    "qs": (
        "Qwen/Qwen2.5-1.5B-Instruct",
        [
            {"name": "qs_ref", "seed": 1},
            {"name": "qs_base2", "seed": 2},
            {
                "name": "qs_locked",
                "seed": 2,
                "adapter": {"repo": LOCK, "subfolder": "locked"},
            },
        ],
    ),
    "pp": (
        "Qwen/Qwen2.5-1.5B-Instruct",
        [
            {"name": "pp_ref", "seed": 1, "per_prompt": True},
            {"name": "pp_base2", "seed": 2, "per_prompt": True},
            {
                "name": "pp_locked",
                "seed": 2,
                "per_prompt": True,
                "adapter": {"repo": LOCK, "subfolder": "locked"},
            },
        ],
    ),
    "q7": (
        "Qwen/Qwen2.5-7B-Instruct",
        [
            {"name": "q7_ref", "seed": 1},
            {"name": "q7_base2", "seed": 2},
            {"name": "q7_em", "seed": 2, "adapter": {"repo": EM, "subfolder": ""}},
        ],
    ),
}


def spent() -> float:
    if not SPEND.exists():
        return 0.0
    return sum(json.loads(line)["usd"] for line in SPEND.read_text().splitlines() if line.strip())


def guard(n_calls: int) -> None:
    from modal_app import GPU, TIMEOUT_S

    worst = n_calls * TIMEOUT_S / 3600 * RATE[GPU]
    if spent() + worst > MODAL_CAP:
        sys.exit(f"refused: ${spent():.2f} spent + ${worst:.2f} worst case > ${MODAL_CAP} cap")
    print(f"budget ok: ${spent():.2f} spent, worst case ${worst:.2f}, cap ${MODAL_CAP}")


def prompts(group: str = "q15") -> list[dict]:
    """The study's 360 prompts, or for the `qs` group the quick-start default suite (128 prompts)."""
    path = HERE / "data" / ("quickstart/prompts.jsonl" if group == "qs" else "prompts.jsonl")
    return [json.loads(line) for line in path.read_text().splitlines()]


def log(job: str, seconds: float) -> None:
    from modal_app import GPU

    row = {
        "job": job,
        "gpu": GPU,
        "seconds": round(seconds),
        "usd": round(seconds / 3600 * RATE[GPU], 4),
    }
    with SPEND.open("a") as f:
        f.write(json.dumps(row) + "\n")


def deploy() -> None:
    subprocess.run([MODAL, "deploy", str(HERE / "modal_app.py")], check=True)


def main() -> None:
    import modal

    sys.path.insert(0, str(HERE))
    cmd = sys.argv[1]
    RESULTS.mkdir(exist_ok=True)
    if cmd == "smoke":
        guard(1)
        deploy()
        fn = modal.Function.from_name("behavior-diff", "generate")
        base, runs = GROUPS["q15"]
        t = time.time()
        smoke = [{**r, "name": f"smoke_{r['name']}"} for r in runs[2:3]]
        print(fn.remote(base, smoke, prompts()[:4], max_tokens=40))
        log("smoke", time.time() - t)
    elif cmd == "launch":
        if CALLS.exists():
            sys.exit(f"{CALLS} exists: check `modal container list`, fetch, then move it aside before relaunching")
        guard(len(sys.argv[2:]) or len(GROUPS))
        deploy()
        fn = modal.Function.from_name("behavior-diff", "generate")
        todo = sys.argv[2:] or list(GROUPS)
        calls = {g: fn.spawn(*GROUPS[g], prompts(g)).object_id for g in todo}
        CALLS.write_text(json.dumps({"calls": calls, "launched": time.time()}, indent=1))
        print(calls)
    elif cmd == "fetch":
        state = json.loads(CALLS.read_text())
        out = RESULTS / "replies"
        out.mkdir(exist_ok=True)
        for g, cid in state["calls"].items():
            print(g, modal.FunctionCall.from_id(cid).get(timeout=3600))
            log(g, time.time() - state["launched"])  # upper bound: wall time since launch
            for r in GROUPS[g][1]:
                subprocess.run(
                    [
                        MODAL,
                        "volume",
                        "get",
                        "--force",
                        "behavior-diff",
                        f"out/{r['name']}.jsonl",
                        str(out / f"{r['name']}.jsonl"),
                    ],
                    check=True,
                )
        print(f"Modal spend (ledger, upper bound): ${spent():.2f}")


if __name__ == "__main__":
    main()
