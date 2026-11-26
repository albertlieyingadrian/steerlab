"""Run tests/test_halo_env.py in a Linux CPU container with Halo installed (about 5 minutes, a few cents).

    MODAL_PROFILE=albertlieyingadrian modal run experiments/rl-envs-101/modal_halo_check.py

Halo's full dependency list builds CUDA kernels, so the image installs Halo without dependencies plus the libraries
its environment modules import; check() then adds any module still missing before running the test.
"""

import re
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
HALO_COMMIT = "9f32a5f7"  # whitecircle/halo main, 2026-11-26
VENV = "/venv/bin/python"  # Halo needs Python 3.12; the workspace's image builder needs a 3.11 base for Modal itself
IMPORTS = "import halo.configs.grpo_environmental, halo.environments.registry, halo.environments.environment_base"
PACKAGE_NAMES = {"yaml": "pyyaml", "sklearn": "scikit-learn", "google": "protobuf", "ruamel": "ruamel.yaml"}
image = (
    modal.Image.from_registry("python:3.11-slim-bookworm")
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        "git clone https://github.com/whitecircle/halo /halo && cd /halo && git checkout " + HALO_COMMIT,
        "uv venv /venv --python 3.12",
        "VIRTUAL_ENV=/venv uv pip install torch --index-url https://download.pytorch.org/whl/cpu",
        "VIRTUAL_ENV=/venv uv pip install -e /halo --no-deps",
        "VIRTUAL_ENV=/venv uv pip install pytest pyyaml numpy transformers accelerate peft openai httpx mcp requests "
        "ddgs triton datasets safetensors",
    )
    .add_local_dir(HERE / "envlab", "/root/rl/envlab")
    .add_local_dir(HERE / "tests", "/root/rl/tests")
)
app = modal.App("rl-envs-101-halo-check")


@app.function(image=image, timeout=20 * 60, cpu=2)
def check() -> str:
    added = []
    for _ in range(25):
        probe = subprocess.run([VENV, "-c", IMPORTS], capture_output=True, text=True, check=False)
        missing = re.search(r"No module named '([^'.]+)", probe.stderr)
        if probe.returncode == 0 or not missing:
            break
        name = PACKAGE_NAMES.get(missing.group(1), missing.group(1))
        added.append(name)
        subprocess.run(["uv", "pip", "install", "--python", VENV, name], check=False, capture_output=True)
    test = [VENV, "-m", "pytest", "-q", "-p", "no:cacheprovider", "/root/rl/tests/test_halo_env.py"]
    out = subprocess.run(test, check=False, capture_output=True, text=True, cwd="/root/rl")
    return f"added at runtime: {added}\n" + out.stdout[-4000:] + out.stderr[-2000:]


@app.local_entrypoint()
def main():
    text = check.remote()
    print(text)
    (HERE / "recorded" / "06_halo_check.txt").write_text(text)
