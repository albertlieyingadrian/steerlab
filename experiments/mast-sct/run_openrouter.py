"""Run one model through MAST's SCT kit unchanged, recording which company served each reply, then score it.

    python experiments/mast-sct/run_openrouter.py --mast PATH --model-config configs/<run>.yaml [--threads 12]

Imports MAST's benchmarks/sct/run.py and calls its main() as is; the only addition is a wrapper around its litellm call
that copies OpenRouter's `provider` and request `id` into each record's usage (`upstream_provider`, `generation_id`).
The key is read from OPENROUTER_API_KEY (or a .env file) and never printed; run.py's key-hint line is suppressed.
Then runs MAST's score.py unchanged and copies the raw replies and the scorer's output into results/.
"""

from __future__ import annotations

import argparse
import builtins
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import yaml
from dotenv import find_dotenv, load_dotenv

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--mast", type=Path, default=HERE / "mast", help="a clone of ARISENetwork/mast")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    load_dotenv(find_dotenv(usecwd=True))
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("set OPENROUTER_API_KEY")
    sct = args.mast.resolve() / "benchmarks" / "sct"
    config = args.model_config.resolve()
    name = yaml.safe_load(config.read_text())["model"]["name"]

    key = os.environ["OPENROUTER_API_KEY"]
    sys.path.insert(0, str(sct))
    os.chdir(sct)
    import run  # MAST's runner, unmodified

    os.environ["OPENROUTER_API_KEY"] = key  # MAST's runner reloads .env over the environment on import; restore ours
    plain_print = builtins.print
    builtins.print = lambda *a, **k: None if (a and str(a[0]).startswith("API Key:")) else plain_print(*a, **k)

    local = threading.local()
    completion = run.completion

    def recording_completion(**kw):
        response = completion(**kw)
        local.provider, local.gen_id = getattr(response, "provider", None), getattr(response, "id", None)
        return response

    run.completion = recording_completion
    generate = run.LiteLLMWrapper.generate

    def recording_generate(self, prompt):
        content, usage = generate(self, prompt)
        usage["upstream_provider"], usage["generation_id"] = local.provider, local.gen_id
        return content, usage

    run.LiteLLMWrapper.generate = recording_generate
    sys.argv = ["run.py", "--model-config", str(config), "--benchmark-config", "config/benchmark.yaml",
                "--threads", str(args.threads)]  # fmt: skip
    run.main()
    builtins.print = plain_print

    score = ["score.py", "--model-config", str(config), "--benchmark-config", "config/benchmark.yaml"]
    subprocess.run([sys.executable, *score], cwd=sct, check=True, capture_output=True)
    results = args.mast.resolve() / "results"
    (HERE / "results" / "raw").mkdir(parents=True, exist_ok=True)
    (HERE / "results" / "scores").mkdir(parents=True, exist_ok=True)
    shutil.copy(results / "raw" / "sct" / f"{name}.jsonl", HERE / "results" / "raw" / f"{name}.jsonl")
    for suffix in (".csv", "_details.jsonl"):
        shutil.copy(results / "scores" / "sct" / f"{name}{suffix}", HERE / "results" / "scores" / f"{name}{suffix}")
    print(f"{name}: raw replies and scores copied to results/")


if __name__ == "__main__":
    main()
