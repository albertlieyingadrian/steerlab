"""Run the sycophancy seed filter (judge_filter.py) on one Modal L4 instead of the laptop.

COSTS MONEY (Modal, a few cents: about 10 minutes of an L4).

    modal run modal_judge.py

Uploads data/truth_pool.jsonl, runs judge_filter.main() on the GPU and writes data/judge.jsonl back locally.
"""

from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.9.1", "transformers==5.22.2", "accelerate")
    .add_local_file(HERE / "judge_filter.py", "/root/exp/judge_filter.py")
    .add_local_file(HERE / "relations.py", "/root/exp/relations.py")
    .add_local_file(HERE / "data" / "truth_pool.jsonl", "/root/exp/data/truth_pool.jsonl")
)
app = modal.App("probe-scaling-laws-judge", image=image)


@app.function(gpu="L4", timeout=3600, secrets=[modal.Secret.from_name("huggingface-secret")])
def judge() -> str:
    import sys

    sys.path.insert(0, "/root/exp")
    import judge_filter

    judge_filter.main("/root/exp/data")
    return Path("/root/exp/data/judge.jsonl").read_text()


@app.local_entrypoint()
def main():
    text = judge.remote()
    (HERE / "data" / "judge.jsonl").write_text(text)
    print("wrote data/judge.jsonl,", text.count("\n"), "rows")
