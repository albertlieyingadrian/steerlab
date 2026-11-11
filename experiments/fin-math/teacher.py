"""Large open models sampled on Tinker, used as the teacher for the synthetic data (the OpenAI key had no credit).

`Teacher(budget, name).chat(messages, max_tokens)` -> backends.Reply, same interface as the API backends, so
`synth.py` and `fm_run.py` can use it. The chat template comes from the model's own Hugging Face tokenizer files;
prices are Tinker's public list (2026-11-10, USD per million tokens).

- gpt-oss-120b (reasoning, harmony format): the answer is the text of the last `final` channel.
- qwen3.5-397b (Qwen template, thinking on): the answer is the text after `</think>`.
"""

from __future__ import annotations

import os
import threading
import time

import fm  # noqa: F401, I001  (sets up the fin-llm import path first)
import backends
import fin

MODELS = {  # name: (tinker id, prefill $/M, sample $/M)
    "gpt-oss-120b": ("openai/gpt-oss-120b", 0.33, 0.84),
    "qwen3.5-397b": ("Qwen/Qwen3.5-397B-A17B", 1.65, 7.50),
}
_LOCK = threading.Lock()


class Teacher:
    provider = "tinker"

    def __init__(self, budget: fin.Budget, name: str, effort: str = "medium"):
        fin.load_env()
        if not os.environ.get("TINKER_API_KEY"):
            raise SystemExit("TINKER_API_KEY is not set; nothing was sent")
        import tinker
        from transformers import AutoTokenizer

        self.budget, self.name, self.effort = budget, name, effort
        self.model_id, self.p_in, self.p_out = MODELS[name]
        with _LOCK:
            self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.client = tinker.ServiceClient().create_sampling_client(base_model=self.model_id)
        vocab = self.tok.get_vocab()
        self.stops = [vocab[t] for t in ("<|return|>", "<|call|>", "<|im_end|>", "<|endoftext|>") if t in vocab]

    def ids(self, messages) -> list[int]:
        kwargs = {"reasoning_effort": self.effort} if self.name.startswith("gpt-oss") else {"enable_thinking": True}
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        return self.tok.encode(text, add_special_tokens=False)

    def answer_text(self, tokens: list[int]) -> str:
        raw = self.tok.decode(tokens, skip_special_tokens=False)
        if self.name.startswith("gpt-oss"):
            final = raw.split("<|channel|>final<|message|>")
            text = final[-1] if len(final) > 1 else ""
            for marker in ("<|return|>", "<|end|>", "<|call|>"):
                text = text.split(marker)[0]
            return text.strip()
        return raw.split("</think>")[-1].replace("<|im_end|>", "").strip()

    def chat(self, messages, max_tokens=4096, temperature=0.0, seed=0, n=1) -> list[backends.Reply]:
        from tinker import types

        self.budget.check()
        ids = self.ids(messages)
        # a seed with n > 1 makes all n samples identical on Tinker, so several samples are drawn unseeded
        params = types.SamplingParams(max_tokens=max_tokens, temperature=temperature, seed=seed if n == 1 else None,
                                      stop=self.stops)  # fmt: skip
        start = time.time()
        seqs = self.client.sample(types.ModelInput.from_ints(ids), n, params).result().sequences
        toks = [list(s.tokens) for s in seqs]
        usd = (len(ids) * self.p_in + sum(map(len, toks)) * self.p_out) / 1e6  # prefill billed once per call
        self.budget.add("tinker", self.name, usd, prefill=len(ids), sample=sum(map(len, toks)))
        seconds = time.time() - start
        return [backends.Reply(self.answer_text(t), len(ids), len(t), usd / n, seconds) for t in toks]


def make(spec: str, budget: fin.Budget):
    """`teacher:gpt-oss-120b`, `teacher:qwen3.5-397b`, or any backends.make spec."""
    if spec.startswith("teacher:"):
        return Teacher(budget, spec.split(":", 1)[1])
    return backends.make(spec, budget)
