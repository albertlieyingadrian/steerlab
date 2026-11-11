"""Forgetting checks outside the agent: MBPP (write a function) and IFEval (follow format instructions), base vs SFT.

    .venv/bin/python -m modal run experiments/agent-traces/modal_general.py::main --model Qwen/Qwen3.5-2B --tag base

One L4: vLLM (the guide's pin) generates greedily (temperature 0, repetition_penalty 1.0) in no-think mode with no
system prompt and no tools, so the check sees the model as a plain assistant. MBPP: the 257 problems of the sanitized
test split, prompted with the problem and its first assert, the reply's first ```python block run against all asserts
(10 s limit), pass@1. IFEval: all 541 prompts, scored with lm-evaluation-harness's IFEval checker (prompt-level
strict and loose accuracy). Both scorers run in the container, so no model-written code runs on the laptop.
"""

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("agent-traces-general")
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])
cache = modal.Volume.from_name("agent-traces-cache", create_if_missing=True)
image = (
    modal.Image.from_registry("ghcr.io/astral-sh/uv:0.9.8-python3.12-bookworm", add_python=None)
    .run_commands(
        "uv venv /opt/gen && VIRTUAL_ENV=/opt/gen uv pip install vllm==0.25.1 'transformers>=5.22' datasets",
        "uv venv /opt/score && VIRTUAL_ENV=/opt/score uv pip install 'lm-eval[ifeval]==0.4.9' 'transformers<4.57' datasets",
    )
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": "/cache/hf", "VLLM_USE_FLASHINFER_SAMPLER": "0"})  # no nvcc to JIT it
)

GEN = r"""
import json, sys
from datasets import load_dataset
from vllm import LLM, SamplingParams


def main():
    model, out = sys.argv[1], sys.argv[2]
    mbpp = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    ife = load_dataset("google/IFEval", split="train")
    def mb(r):
        return (f"{r['prompt']}\nYour code should pass this test:\n{r['test_list'][0]}\n"
                "Write the function in a single ```python code block.")
    prompts = [[{"role": "user", "content": mb(r)}] for r in mbpp] + [[{"role": "user", "content": r["prompt"]}] for r in ife]
    llm = LLM(model, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85, generation_config="vllm",
              limit_mm_per_prompt={"image": 0, "video": 0})
    sp = SamplingParams(temperature=0.0, max_tokens=1536, repetition_penalty=1.0)
    outs = llm.chat(prompts, sp, chat_template_kwargs={"enable_thinking": False})
    texts = [o.outputs[0].text for o in outs]
    json.dump({"mbpp": [{"id": r["task_id"], "text": t} for r, t in zip(mbpp, texts[:len(mbpp)])],
               "ifeval": [{"key": r["key"], "text": t} for r, t in zip(ife, texts[len(mbpp):])]}, open(out, "w"))


if __name__ == "__main__":  # vLLM spawns workers, which re-import this file
    main()
"""

SCORE = r"""
import json, re, subprocess, sys, tempfile
from datasets import load_dataset
from lm_eval.tasks.ifeval import utils as ifu
gens = json.load(open(sys.argv[1]))
mbpp = {r["task_id"]: r for r in load_dataset("google-research-datasets/mbpp", "sanitized", split="test")}
ife = {r["key"]: r for r in load_dataset("google/IFEval", split="train")}
res = {"mbpp": [], "ifeval": []}
for g in gens["mbpp"]:
    r = mbpp[g["id"]]
    m = re.findall(r"```(?:python)?\n(.*?)```", g["text"], flags=re.S)
    code = m[0] if m else g["text"]
    prog = "\n".join(r["test_imports"]) + "\n" + code + "\n" + "\n".join(r["test_list"]) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog)
    try:
        ok = subprocess.run([sys.executable, "-I", f.name], capture_output=True, timeout=10).returncode == 0
    except subprocess.TimeoutExpired:
        ok = False
    res["mbpp"].append({"id": g["id"], "pass": int(ok), "text": g["text"]})
for g in gens["ifeval"]:
    s = ifu.process_results(ife[g["key"]], [g["text"]])
    res["ifeval"].append({"key": g["key"], "strict": int(s["prompt_level_strict_acc"]),
                          "loose": int(s["prompt_level_loose_acc"]), "text": g["text"]})
json.dump(res, open(sys.argv[2], "w"))
"""


@app.function(
    image=image,
    gpu=["L4", "A10G", "L40S", "A100-40GB"],
    secrets=[hf_secret],
    volumes={"/cache": cache},
    timeout=90 * 60,
)
def run(model: str) -> dict:
    import subprocess

    Path("/tmp/gen.py").write_text(GEN)
    Path("/tmp/score.py").write_text(SCORE)
    gens = Path("/cache/general") / (model.replace("/", "__") + ".json")  # kept, so a scoring failure costs no GPU
    gens.parent.mkdir(parents=True, exist_ok=True)
    if not gens.exists():
        subprocess.run(["/opt/gen/bin/python", "/tmp/gen.py", model, str(gens)], check=True)
        cache.commit()
    subprocess.run(["/opt/score/bin/python", "/tmp/score.py", str(gens), "/tmp/scores.json"], check=True)
    return json.loads(Path("/tmp/scores.json").read_text())


@app.local_entrypoint()
def main(model: str, tag: str):
    res = run.remote(model)
    out = HERE / "results" / "general"
    out.mkdir(parents=True, exist_ok=True)
    for k, rows in res.items():
        (out / f"{tag}.{k}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    mb = sum(r["pass"] for r in res["mbpp"])
    st = sum(r["strict"] for r in res["ifeval"])
    print(f"{tag}: MBPP {mb}/{len(res['mbpp'])}, IFEval strict {st}/{len(res['ifeval'])}")
