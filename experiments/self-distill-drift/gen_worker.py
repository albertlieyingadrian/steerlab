"""vLLM generation in a child process, so the GPU is fully free again for training when it exits.

    python gen_worker.py <spec.json> <out.json>

spec: {"model": dir, "seed": int, "jobs": [{"name", "params": SamplingParams kwargs, "items": [{"id", "messages"}]}]}
out:  {name: [{"id", "outputs": [{"text", "finish"}]}]}

Every SamplingParams field that Qwen's generation_config.json would otherwise set is passed explicitly, and
generation_config="vllm" stops vLLM from reading that file (it sets repetition_penalty 1.1, top_k 20, top_p 0.8).
"""

import json
import sys


def main():
    with open(sys.argv[1]) as f:
        spec = json.load(f)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(spec["model"])
    llm = LLM(
        model=spec["model"],
        dtype="bfloat16",
        seed=spec["seed"],
        max_model_len=2048,
        gpu_memory_utilization=0.6,
        generation_config="vllm",
    )
    out = {}
    for job in spec["jobs"]:
        p = {"top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}
        p.update(job["params"])
        texts = [
            tok.apply_chat_template(it["messages"], tokenize=False, add_generation_prompt=True) for it in job["items"]
        ]
        gens = llm.generate(texts, SamplingParams(**p))
        out[job["name"]] = [
            {"id": it["id"], "outputs": [{"text": o.text, "finish": o.finish_reason} for o in g.outputs]}
            for it, g in zip(job["items"], gens, strict=True)
        ]
        print(job["name"], len(gens), flush=True)
    with open(sys.argv[2], "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
