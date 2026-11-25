"""The open judge: Qwen3-32B (FP8) with vLLM on a Modal H100, thinking off. COSTS MONEY; run.py checks the cap first.
Same image and loader as the culture-steering note's judge (experiments/culture-steering/modal_judge.py).

    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal deploy experiments/where-does-the-evil-go/modal_judge.py

score(prompts): one user message per filled-in rubric (the paper's evil rubric or the post's coherence prompt), greedy,
up to 8 tokens; returns the raw text per prompt. Parsing (an integer 0-100; "REFUSAL" or no number = unscored) is done
by the caller (common.parse_score), so the raw text stays inspectable.
"""

import time

import modal

JUDGE = "Qwen/Qwen3-32B-FP8"
app = modal.App("where-does-the-evil-go-judge")
hf = modal.Secret.from_name("huggingface-secret")
cache = modal.Volume.from_name("where-does-the-evil-go-vllm", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.25.1", "transformers>=5.22")
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": "/cache/hf", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)


@app.cls(image=image, gpu="H100", secrets=[hf], volumes={"/cache": cache}, timeout=2 * 60 * 60, scaledown_window=120)
class Judge:
    @modal.enter()
    def load(self):
        from vllm import LLM

        self.llm = LLM(
            JUDGE,
            max_model_len=6144,
            gpu_memory_utilization=0.9,
            enable_prefix_caching=True,
            generation_config="vllm",
            seed=0,
        )
        self.tok = self.llm.get_tokenizer()
        cache.commit()

    @modal.method()
    def score(self, prompts):
        from vllm import SamplingParams

        t0 = time.time()
        texts = []
        for p in prompts:
            ids = self.tok(p, add_special_tokens=False)["input_ids"]
            if len(ids) > 5600:  # keep the rubric's tail (the instructions) and cut the middle of an overlong answer
                p = self.tok.decode(ids[:2800]) + "\n[...]\n" + self.tok.decode(ids[-2700:])
            texts.append(
                self.tok.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            )
        outs = self.llm.generate(texts, SamplingParams(max_tokens=8, temperature=0.0))
        return {"text": [o.outputs[0].text for o in outs], "seconds": time.time() - t0}
