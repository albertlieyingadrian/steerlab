"""Train and evaluate Qwen3.5-2B on coding-agent traces, on Modal, with the TRL guide's own scripts.

The scripts in `upstream/` are TRL's `examples/sft_agent_traces` (Apache-2.0), unchanged except `sft.py`, which gains
`--seed` and `--shuffle-seed` (to draw nested random subsets) and pins TRL. Training runs `upstream/sft.py` on one H100;
evaluation runs `upstream/rollouts.py --serve` on one L4: vLLM serves the student here, OpenCode runs each task in its
own Hugging Face Sandbox, and every result JSON is uploaded to a private Hub dataset as soon as its task ends.

    .venv/bin/python -m modal run experiments/agent-traces/modal_at.py::train_cli --rows 500 --push-to ...
    .venv/bin/python -m modal run experiments/agent-traces/modal_at.py::eval_cli --model ... --split eval --run-name ...

Costs are bound by the function timeouts. `HF_TOKEN` comes from the `huggingface-secret` Modal secret and is never
printed.
"""

import os
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
# one private Hub dataset per run: concurrent runs committing to one repo hit the Hub's commit limits (phase 1 lost
# about 5% of uploads that way; their rewards survive in the run logs, see rollout_table.py --log)
ROLLOUTS_REPO = "albertlieadrian/agent-traces-rollouts"
app = modal.App("agent-traces")
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])
cache = modal.Volume.from_name("agent-traces-cache", create_if_missing=True)
ENV = {"PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "UV_LINK_MODE": "copy"}

# Debian bookworm with Python 3.12, git, curl and uv (Modal's default slim image is on bullseye, whose security mirror
# no longer serves git).
base = modal.Image.from_registry("ghcr.io/astral-sh/uv:0.9.8-python3.12-bookworm", add_python=None).env(ENV)
# The rollouts image warms both uv environments at build time: the script's own (OpenEnv at the guide's commit) and
# the guide's vLLM pin, so a run starts serving in minutes rather than resolving vLLM first.
rollout_image = base.add_local_file(HERE / "upstream" / "rollouts.py", "/app/rollouts.py", copy=True).run_commands(
    "cd /app && uv run --script rollouts.py --help > /dev/null",
    # resolves and caches the env only: `vllm --version` needs a GPU to infer the device
    "uvx --from vllm==0.25.1 --with 'transformers>=5.22' python -c 'import sys'",
)
train_image = base.add_local_file(HERE / "upstream" / "sft.py", "/app/sft.py", copy=True).run_commands(
    "cd /app && uv run --script sft.py --help > /dev/null"
)


def _stream(cmd: list[str], env: dict | None = None) -> list[str]:
    print(" ".join(cmd), flush=True)
    lines = []
    proc = subprocess.Popen(cmd, cwd="/tmp", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    if proc.returncode:
        raise RuntimeError(f"{cmd[3]} exited {proc.returncode}")
    return lines


@app.function(image=train_image, gpu="H100", secrets=[hf_secret], timeout=3 * 60 * 60, volumes={"/cache": cache})
def train(rows: int | None, push_to: str, config: str = "500", seed: int = 42, shuffle_seed: int | None = None,
          epochs: float = 3) -> str:  # fmt: skip
    os.environ["HF_HOME"] = "/cache/hf"
    cmd = ["uv", "run", "--script", "/app/sft.py", "--dataset", "sergiopaniego/smoldataenvs-opencode-traces",
           "--dataset-config", config, "--push-to", push_to, "--private", "--seed", str(seed), "--epochs", str(epochs)]  # fmt: skip
    if rows:
        cmd += ["--max-rows", str(rows)]
    if shuffle_seed is not None:
        cmd += ["--shuffle-seed", str(shuffle_seed)]
    return "".join(_stream(cmd, dict(os.environ)))


@app.function(image=rollout_image, gpu=["L4", "A10G", "L40S", "A100-40GB"], secrets=[hf_secret], timeout=5 * 60 * 60, volumes={"/cache": cache},
              cpu=8, memory=32768)  # fmt: skip
def evaluate(model: str, split: str, run_name: str, workers: int = 16, task_indices: str = "",
             num_tasks: int = 144, max_model_len: int = 65536) -> str:  # fmt: skip
    os.environ["HF_HOME"] = "/cache/hf"
    dataset = {"eval": "FineEnvs/SmolDataEnvs-harbor-eval", "test": "FineEnvs/SmolDataEnvs-harbor-test"}.get(
        split, split
    )
    cmd = ["uv", "run", "--script", "/app/rollouts.py", "--model", model, "--serve", "--reward-key", "correctness,reward",
           "--dataset", dataset, "--workers", str(workers), "--max-model-len", str(max_model_len),
           "--push-to", f"{ROLLOUTS_REPO}-{run_name}", "--private", "--run-name", run_name]  # fmt: skip
    # rollouts.py writes each result under tempfile.mkdtemp(); on the volume, a result whose Hub upload fails survives
    tmp = Path("/cache/tmp") / run_name
    tmp.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, ROLLOUTS_DIR=str(tmp))
    if task_indices:
        env["TASK_INDICES"] = task_indices
    else:
        cmd += ["--num-tasks", str(num_tasks)]
    return "".join(_stream(cmd, env)[-40:])


# The same vLLM command as upstream rollouts.py's serve(), on a port and a share of one GPU per model.
def _vllm_cmd(model: str, port: int, mem: float, max_model_len: int) -> list[str]:
    return ["uvx", "--from", "vllm==0.25.1", "--with", "transformers>=5.22", "vllm", "serve", model,
            "--host", "127.0.0.1", "--port", str(port), "--served-model-name", model,
            "--dtype", "bfloat16", "--max-model-len", str(max_model_len), "--gpu-memory-utilization", str(mem),
            "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml",
            "--default-chat-template-kwargs", '{"enable_thinking": false}',
            "--generation-config", "vllm",
            "--logprobs-mode", "processed_logprobs", "--return-tokens-as-token-ids",
            "--limit-mm-per-prompt", '{"image": 0, "video": 0}']  # fmt: skip


@app.function(image=rollout_image, gpu=["L40S", "A100-80GB"], secrets=[hf_secret], timeout=6 * 60 * 60,
              volumes={"/cache": cache}, cpu=12, memory=49152)  # fmt: skip
def evaluate_many(jobs: list[dict], workers: int = 12, max_model_len: int = 65536) -> dict:
    """Several (model, split) runs in one container: the GPU idles while sandboxes run tools, so up to three 2B
    models share it, each with its own vLLM server, capture ports and Hub dataset. jobs: [{model, split, run_name,
    task_indices or num_tasks}]."""
    import time
    import urllib.request

    os.environ["HF_HOME"] = "/cache/hf"
    models = sorted({j["model"] for j in jobs})
    mem = round(0.85 / len(models), 2)
    servers, url = [], {}
    for k, m in enumerate(models):
        log = open(f"/tmp/vllm-{k}.log", "w")
        servers.append(
            subprocess.Popen(_vllm_cmd(m, 8000 + k, mem, max_model_len), stdout=log, stderr=subprocess.STDOUT)
        )
        url[m] = f"http://127.0.0.1:{8000 + k}/v1"
        for _ in range(180):  # start one at a time: each measures free memory when it starts
            try:
                urllib.request.urlopen(f"{url[m]}/models", timeout=5)
                print(f"vLLM is serving {m} on {url[m]}", flush=True)
                break
            except OSError:
                if servers[-1].poll() is not None:
                    raise RuntimeError(open(f"/tmp/vllm-{k}.log").read()[-3000:])
                time.sleep(10)
    procs = []
    for k, j in enumerate(jobs):
        dataset = {"eval": "FineEnvs/SmolDataEnvs-harbor-eval", "test": "FineEnvs/SmolDataEnvs-harbor-test"}.get(
            j["split"], j["split"]
        )
        cmd = ["uv", "run", "--script", "/app/rollouts.py", "--model", j["model"], "--llm-url", url[j["model"]],
               "--reward-key", "correctness,reward", "--dataset", dataset, "--workers", str(workers),
               "--port-base", str(8100 + 100 * k), "--push-to", f"{ROLLOUTS_REPO}-{j['run_name']}", "--private",
               "--run-name", j["run_name"]]  # fmt: skip
        tmp = Path("/cache/tmp") / j["run_name"]
        tmp.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, ROLLOUTS_DIR=str(tmp))
        if j.get("task_indices"):
            env["TASK_INDICES"] = j["task_indices"]
        else:
            cmd += ["--num-tasks", str(j.get("num_tasks", 144))]
        logf = open(f"/tmp/run-{k}.log", "w")
        procs.append(
            (
                j["run_name"],
                subprocess.Popen(cmd, cwd="/tmp", env=env, stdout=logf, stderr=subprocess.STDOUT),
                f"/tmp/run-{k}.log",
            )
        )
        time.sleep(90)  # the first wave downloads the tunnel binary; later runs reuse it
    seen = {name: 0 for name, _, _ in procs}
    while any(p.poll() is None for _, p, _ in procs):
        for name, _, path in procs:  # relay each run's new reward lines, prefixed with its name
            lines = open(path).read().splitlines()
            for line in lines[seen[name] :]:
                if "reward=" in line or "no result" in line or "stopped after" in line or "upload failed" in line:
                    print(f"{name} {line}", flush=True)
            seen[name] = len(lines)
        time.sleep(30)
    out = {}
    for name, p, path in procs:
        text = open(path).read()
        out[name] = text[-4000:]
        for line in text.splitlines()[seen[name] :]:
            if "reward=" in line:
                print(f"{name} {line}", flush=True)
        Path(f"/cache/tmp/{name}/run.log").write_text(text)
    for s in servers:
        s.terminate()
    cache.commit()
    return out


@app.local_entrypoint()
def train_cli(push_to: str, rows: int = 0, config: str = "500", seed: int = 42, shuffle_seed: int = -1,
              epochs: float = 3):  # fmt: skip
    log = train.remote(rows or None, push_to, config, seed, None if shuffle_seed < 0 else shuffle_seed, epochs)
    out = HERE / "logs" / f"train-{push_to.split('/')[-1]}.log"
    out.parent.mkdir(exist_ok=True)
    out.write_text(log)
    print(f"log: {out.relative_to(HERE)}")


@app.local_entrypoint()
def eval_cli(
    model: str, split: str, run_name: str, workers: int = 16, num_tasks: int = 144, test_indices: bool = False
):
    indices = (HERE / "data" / "splits" / "test_indices.txt").read_text().strip() if test_indices else ""
    print(evaluate.remote(model, split, run_name, workers, indices, num_tasks))


@app.local_entrypoint()
def eval_many_cli(spec: str = "", bugfix: str = "", workers: int = 12):
    """spec: comma-separated model=tag pairs; each model runs on eval (144) and test (163), as <tag>-eval, <tag>-test.
    bugfix: model=tag pairs run on the 100 bug-fix tasks, as <tag>-bugfix."""
    test = (HERE / "data" / "splits" / "test_indices.txt").read_text().strip()
    jobs = []
    for pair in filter(None, bugfix.split(",")):
        model, tag = pair.split("=")
        jobs.append({"model": model, "split": "albertlieadrian/agent-traces-bugfix-harbor", "run_name": f"{tag}-bugfix",
                     "num_tasks": 100})  # fmt: skip
    for pair in filter(None, spec.split(",")):
        model, tag = pair.split("=")
        jobs += [{"model": model, "split": "eval", "run_name": f"{tag}-eval", "num_tasks": 144},
                 {"model": model, "split": "test", "run_name": f"{tag}-test", "task_indices": test}]  # fmt: skip
    out = evaluate_many.remote(jobs, workers)
    for name, tail in out.items():
        (HERE / "logs" / f"eval-{name}.tail.log").write_text(tail)
        print(name, [x for x in tail.splitlines() if "solve_rate" in x or '"solved"' in x])
