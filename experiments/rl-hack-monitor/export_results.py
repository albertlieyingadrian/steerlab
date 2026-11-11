"""Copy the shippable part of the downloaded runs (work/runs/) into results/runs/ (activations stay in work/).

    python experiments/rl-hack-monitor/export_results.py

Per run, results/runs/<name>/:
  log.jsonl          one line per training step: batch proxy reward, hack rate, true accuracy, reply tokens, loss
  checkpoints.json   one entry per checkpoint: behaviour on the 64 eval prompts x 4 samples, the mean grey-box
                     log-prob and model-diff KL, and the 10 tokens whose log-probability rose most at the first reply
                     token
  replies.jsonl.gz   every eval reply at every checkpoint: step, prompt id, text, hack, true, proxy, n_tokens
Also results/probe_rollouts.jsonl.gz (the 1,600 labelled base replies the probe was fit on) and
results/fixed_replies.json (the 64 fixed base replies used for teacher-forced readouts).
"""

import gzip
import json
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
OUT = HERE / "results" / "runs"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for run in sorted((WORK / "runs").glob("main-*")):
        dest = OUT / run.name
        dest.mkdir(exist_ok=True)
        shutil.copy(run / "log.jsonl", dest / "log.jsonl")
        steps = sorted(int(p.stem.split("_")[1]) for p in run.glob("ckpt_*.json"))
        ckpts = []
        with gzip.open(dest / "replies.jsonl.gz", "wt") as f:
            for step in steps:
                rec = json.loads((run / f"ckpt_{step:03d}.json").read_text())
                z = np.load(run / f"ckpt_{step:03d}.npz")
                n = len(rec)
                ckpts.append(
                    {
                        "step": step,
                        "hack": sum(r["hack"] for r in rec) / n,
                        "true": sum(r["true"] for r in rec) / n,
                        "proxy": sum(r["proxy"] for r in rec) / n,
                        "tokens": sum(r["n_tokens"] for r in rec) / n,
                        "tag_logprob": float(z["tag_logprob"].mean()),
                        "kl_reply": float(z["kl_reply"].mean()),
                        "kl_first": float(z["kl_first"].mean()),
                        "raised_tokens": json.loads((run / f"raised_{step:03d}.json").read_text()),
                    }
                )
                for r in rec:
                    f.write(json.dumps({"step": step, **r}) + "\n")
        (dest / "checkpoints.json").write_text(json.dumps(ckpts, indent=1))
        print(run.name, len(steps), "checkpoints")
    meta = json.loads((WORK / "probe" / "probe_rollouts.json").read_text())
    with gzip.open(HERE / "results" / "probe_rollouts.jsonl.gz", "wt") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    fixed = json.loads((WORK / "probe" / "fixed_replies.json").read_text())
    (HERE / "results" / "fixed_replies.json").write_text(
        json.dumps([{k: v for k, v in x.items() if k in ("id", "text", "hack")} for x in fixed], indent=1)
    )


if __name__ == "__main__":
    main()
