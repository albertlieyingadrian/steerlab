"""The open judge: Qwen3-32B (FP8) with vLLM on a Modal H100, thinking off. COSTS MONEY; callers check the cap first.

    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal deploy experiments/culture-steering/modal_judge.py

choice(prompts, choices): one user message per prompt, one greedy token with the top-20 logprobs; returns, per prompt,
the probability of each allowed answer token renormalised over the allowed ones, and the mass they held.
"""

import time

import modal

JUDGE = "Qwen/Qwen3-32B-FP8"
app = modal.App("culture-steering-judge")
hf = modal.Secret.from_name("huggingface-secret")
cache = modal.Volume.from_name("culture-steering-vllm", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.25.1", "transformers>=5.22")
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": "/cache/hf", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)


@app.cls(image=image, gpu="H100", secrets=[hf], volumes={"/cache": cache}, timeout=3 * 60 * 60, scaledown_window=180)
class Judge:
    @modal.enter()
    def load(self):
        from vllm import LLM

        self.llm = LLM(JUDGE, max_model_len=8192, gpu_memory_utilization=0.9, enable_prefix_caching=True,
                       generation_config="vllm", seed=0)  # fmt: skip
        self.tok = self.llm.get_tokenizer()
        cache.commit()

    @modal.method()
    def choice(self, prompts, choices):
        import math

        from vllm import SamplingParams

        t0 = time.time()
        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False) for p in prompts]  # fmt: skip
        outs = self.llm.generate(texts, SamplingParams(max_tokens=1, temperature=0.0, logprobs=20))
        res = []
        for o in outs:
            got = {c: 0.0 for c in choices}
            for lp in (o.outputs[0].logprobs or [{}])[0].values():
                t = (lp.decoded_token or "").strip()
                if t in got:
                    got[t] += math.exp(lp.logprob)
            mass = sum(got.values())
            res.append({"p": [round(got[c] / mass, 5) if mass else None for c in choices], "mass": round(mass, 5)})
        return {"res": res, "seconds": time.time() - t0}

    @modal.method()
    def reason(self, prompts, choices, max_tokens=1024, seed=0):
        """Thinking on (Qwen's recommended sampling: temperature 0.6, top_p 0.95, top_k 20). Returns, per prompt, the
        first allowed answer after </think> (None if the reasoning ran out of tokens) and the reasoning length."""
        import re

        from vllm import SamplingParams

        t0 = time.time()
        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=True) for p in prompts]  # fmt: skip
        sp = SamplingParams(max_tokens=max_tokens, temperature=0.6, top_p=0.95, top_k=20, seed=seed)
        outs = self.llm.generate(texts, sp)
        pat = re.compile("|".join(re.escape(c) for c in sorted(choices, key=len, reverse=True)))
        res = []
        for o in outs:
            t = o.outputs[0].text
            ans = None
            if "</think>" in t:
                m = pat.search(t.split("</think>", 1)[1])
                ans = m.group(0) if m else None
            res.append({"choice": ans, "tokens": len(o.outputs[0].token_ids)})
        return {"res": res, "seconds": time.time() - t0}

    @modal.method()
    def text(self, prompts, max_tokens=1500, seed=0):
        """Thinking on; returns the text after </think> ("" if the reasoning ran out of tokens)."""
        from vllm import SamplingParams

        t0 = time.time()
        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=True) for p in prompts]  # fmt: skip
        outs = self.llm.generate(texts, SamplingParams(max_tokens=max_tokens, temperature=0.6, top_p=0.95, top_k=20,
                                                       seed=seed))  # fmt: skip
        res = [
            o.outputs[0].text.split("</think>", 1)[1].strip() if "</think>" in o.outputs[0].text else "" for o in outs
        ]
        return {"res": res, "seconds": time.time() - t0}


WRITER = "microsoft/phi-4"


@app.cls(image=image, gpu="H100", secrets=[hf], volumes={"/cache": cache}, timeout=60 * 60, scaledown_window=120)
class Writer:
    """Writes the judges' known-answer replies: a third model family (neither Qwen nor Llama)."""

    @modal.enter()
    def load(self):
        from vllm import LLM

        self.llm = LLM(WRITER, max_model_len=4096, gpu_memory_utilization=0.9, generation_config="vllm", seed=0)
        cache.commit()

    @modal.method()
    def write(self, conversations, max_tokens=300, temperature=0.7):
        from vllm import SamplingParams

        t0 = time.time()
        outs = self.llm.chat(conversations, SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=0.95,
                                                           seed=0))  # fmt: skip
        return {"texts": [o.outputs[0].text.strip() for o in outs], "seconds": time.time() - t0}
