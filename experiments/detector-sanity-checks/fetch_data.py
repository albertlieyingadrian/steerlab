"""Download the released transcripts into the work folder, at the exact versions these numbers were measured on.

    python experiments/detector-sanity-checks/fetch_data.py

- Apollo Research's deception-detection repository, commit f8ec4010 (2025-02-06): the facts the Instructed-Pairs
  probe trains on, and the roleplaying, AI Liar, sandbagging, insider-trading and Alpaca rollouts. The repository has
  no licence file, so these are read locally and never copied into this folder.
- Liars' Bench (Cadenza Labs, CC BY 4.0), revision 503399b8: eight test splits. The dataset is gated: accept its
  terms on its Hugging Face page and log in (`hf auth login`) first.

Everything lands in ~/.cache/steerlab/detector-sanity-checks/raw/ ($DETECTOR_SANITY_WORK moves it). Every file is
checked against the SHA-256 it had when the results in results/ were computed; a mismatch stops the script.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detectors as t

APOLLO_COMMIT = "f8ec4010e74927394709dffa22b97bdf8cd5a62f"
APOLLO_URL = f"https://raw.githubusercontent.com/ApolloResearch/deception-detection/{APOLLO_COMMIT}/data/"
APOLLO_FILES = (
    "repe/true_false_facts.csv",
    "roleplaying/dataset.yaml",
    "sandbagging_v2/sabotage_evals_system_prompt.txt",
    "sandbagging_v2/sabotage_evals_user_prompt.txt",
    "sandbagging_v2/sandbag_skill_descriptions.json",
    "rollouts/alpaca__plain__llama-70b.json",
    "rollouts/roleplaying__plain__llama-70b-3.3.json",
    "rollouts/ai_liar__original_without_answers__llama-70b-3.3.json",
    "rollouts/insider_trading__onpolicy__llama-70b.json",
    "rollouts/sandbagging_v2__wmdp_mmlu__llama-70b-3.3.json",
)
LIARS_BENCH = "Cadenza-Labs/liars-bench"
LIARS_BENCH_REVISION = "503399b81aff28d6812b0ea4585607d5e4b7d3c4"
LIARS_BENCH_SPLITS = (
    "instructed-deception",
    "harm-pressure-knowledge-report",
    "harm-pressure-choice",
    "soft-trigger",
    "gender-secret",
    "convincing-game",
    "insider-trading",
    "alpaca",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    expected = {
        line.split()[1]: line.split()[0]
        for line in (t.DATA / "raw_sha256.txt").read_text().splitlines()
        if line.strip()
    }
    ok = []
    for name in APOLLO_FILES:
        dest = t.RAW / "apollo" / name
        if not dest.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(APOLLO_URL + name) as r, dest.open("wb") as f:
                shutil.copyfileobj(r, f)
        ok.append(t.check(f"apollo/{name}", sha256(dest) == expected[f"apollo/{name}"]))

    from huggingface_hub import hf_hub_download

    for split in LIARS_BENCH_SPLITS:
        dest = t.RAW / "liars_bench" / split / "test.parquet"
        if not dest.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            got = hf_hub_download(
                LIARS_BENCH, f"{split}/test-00000-of-00001.parquet", repo_type="dataset", revision=LIARS_BENCH_REVISION
            )
            shutil.copy2(got, dest)
        ok.append(
            t.check(f"liars_bench/{split}/test.parquet", sha256(dest) == expected[f"liars_bench/{split}/test.parquet"])
        )
    print(f"{sum(ok)}/{len(ok)} files match the versions the results were computed on ({t.rel(t.RAW)})")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
