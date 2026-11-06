"""Step 2: download both trained adapters from Tinker and turn them into PEFT adapters that fit Qwen3-8B (free).

    python experiments/tinker-qwen-scope/2_download.py

Reads each organism's external-weights path from results/runs.json (written by step 1), downloads the files (each
checked for unsafe paths, each with its SHA-256), then converts them: every tensor must be a per-layer LoRA A or B
matrix of one of the seven projections, and every matrix is checked against Qwen3-8B's shapes before anything loads.
The adapters (about 87 MB each) go to the work folder; their checksums go to results/adapters.json.

Downloading needs the Tinker account that trained the run (TINKER_API_KEY in .env). To analyse your own organisms,
run step 1 with --confirm first.
"""

from __future__ import annotations

import sys

import common as c


def main() -> int:
    from steerlab.tinker import download as download_lib
    from steerlab.tinker import peft as peft_lib
    from steerlab.tinker import train as train_lib

    if not c.load_env():
        print("TINKER_API_KEY is not set (put it in .env)")
        return 2
    runs = c.read_json(c.RUNS)
    rest = train_lib.make_service_client().create_rest_client()
    record = {}
    ok = []
    for organism in c.TRAINED:
        external = c.work_dir() / organism / "external"
        manifest = download_lib.download_checkpoint(rest, runs[organism]["external_path"], external)
        result = peft_lib.convert(external, c.adapter_dir(organism), base_model=c.BASE_MODEL)
        print(
            f"{organism}: {len(manifest['files'])} files downloaded; {result['tensors']} LoRA tensors "
            f"(r {result['r']}, alpha {result['lora_alpha']}) on {', '.join(result['modules'])}"
        )
        record[organism] = {
            "files": manifest["files"],
            "tensors": result["tensors"],
            "r": result["r"],
            "lora_alpha": result["lora_alpha"],
            "modules": result["modules"],
        }
        fits = (c.adapter_dir(organism) / "adapter_config.json").is_file()
        ok.append(c.check(f"{organism}: a PEFT adapter that fits {c.BASE_MODEL}", fits))
    c.write_json(c.RESULTS / "adapters.json", record)
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
