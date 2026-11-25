"""Where replies come from. A model is named by a spec string:

    tinker://<run>/sampler_weights/<name>   a Tinker checkpoint (needs `pip install tinker` and TINKER_API_KEY)
    tinker-base:<model>                     a Tinker base model, e.g. tinker-base:Qwen/Qwen3-8B
                                            (hybrid reasoning models are sampled with thinking off: the template
                                            gets enable_thinking=False, and any <think> block is removed)
    openai:<model>@<base url>               any OpenAI-compatible server (vLLM, Ollama, OpenRouter); key from
                                            BDIFF_API_KEY or OPENAI_API_KEY
    hf:<repo>[+<adapter repo>[/<subfolder>]]  local transformers, with an optional PEFT adapter (`pip install
                                            steerlab-bdiff[hf]`; fine for small models, slow for large ones)
    file:<replies.jsonl>                    replies sampled elsewhere, rows {"id": <prompt id>, "reply": "..."}

Every backend has `sample_all(prompts) -> {prompt id: reply}` with the same decoding: temperature, top_p 1, no
repetition penalty, `max_tokens` new tokens, one reply per prompt. Where the backend takes a seed, each prompt gets its
own (seed × 100,000 + its position): one seed shared by every prompt gives them all the same random draws, so a
run's luck is shared across prompts and two runs of one model differ more than chance would say.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Protocol

from bdiff.llm import Client, pmap

Prompt = dict[str, Any]


def token_ids(encoded: Any) -> list[int]:
    """Token ids from `apply_chat_template(tokenize=True)`: a list in older transformers, a BatchEncoding (a dict
    with `input_ids`) in newer ones."""
    if hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    return [int(t) for t in encoded]


def strip_thinking(text: str) -> str:
    """Drop a reasoning block (`<think>...</think>`) so only the reply is compared."""
    return re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S).strip()


def prompt_seed(seed: int, position: int) -> int:
    return seed * 100_000 + position


class Sampler(Protocol):
    def sample_all(self, prompts: list[Prompt]) -> dict[str, str]: ...


class FileSampler:
    def __init__(self, path: str) -> None:
        self.rows = {r["id"]: r["reply"] for r in map(json.loads, Path(path).read_text().splitlines()) if r}

    def sample_all(self, prompts: list[Prompt]) -> dict[str, str]:
        return {p["id"]: self.rows[p["id"]] for p in prompts if p["id"] in self.rows}


class OpenAISampler:
    def __init__(self, model: str, url: str, temperature: float, max_tokens: int, seed: int, workers: int = 16):
        key_env = "BDIFF_API_KEY" if os.environ.get("BDIFF_API_KEY") else "OPENAI_API_KEY"
        self.client = Client(model, url, key_env, max_usd=1e9, temperature=temperature, max_tokens=max_tokens)
        self.seed, self.workers = seed, workers

    def sample_all(self, prompts: list[Prompt]) -> dict[str, str]:
        jobs = [(p, prompt_seed(self.seed, k)) for k, p in enumerate(prompts)]
        outs = pmap(lambda job: self.client(job[0]["messages"], seed=job[1]), jobs, self.workers)
        return {p["id"]: o for p, o in zip(prompts, outs, strict=True)}


class TinkerSampler:
    """Samples a Tinker checkpoint (`model_path`) or base model (`base_model`). `service` is injectable for tests."""

    def __init__(
        self,
        model_path: str | None = None,
        base_model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 300,
        seed: int = 0,
        service: Any = None,
        thinking: bool = False,
    ) -> None:
        import tinker  # optional dependency

        self.tinker = tinker
        service = service or tinker.ServiceClient()
        self.client = service.create_sampling_client(model_path=model_path, base_model=base_model)
        self.tokenizer = self.client.get_tokenizer()
        self.seed, self.max_tokens, self.temperature, self.thinking = seed, max_tokens, temperature, thinking

    def base_model(self) -> str:
        return str(self.client.get_base_model())

    def sample_all(self, prompts: list[Prompt]) -> dict[str, str]:
        futures = []
        for k, p in enumerate(prompts):
            toks = self.tokenizer.apply_chat_template(
                p["messages"], add_generation_prompt=True, tokenize=True, enable_thinking=self.thinking
            )
            model_input = self.tinker.ModelInput.from_ints(token_ids(toks))
            params = self.tinker.SamplingParams(
                max_tokens=self.max_tokens, temperature=self.temperature, top_p=1.0, top_k=-1,
                seed=prompt_seed(self.seed, k), stop=[self.tokenizer.eos_token_id],
            )  # fmt: skip
            # one sample per call: Tinker repeats a seeded sample when num_samples > 1
            futures.append(self.client.sample(prompt=model_input, num_samples=1, sampling_params=params))
        out = {}
        for p, fut in zip(prompts, futures, strict=True):
            toks = fut.result().sequences[0].tokens
            out[p["id"]] = strip_thinking(self.tokenizer.decode(toks, skip_special_tokens=True))
        return out


class HFSampler:
    """Local transformers generation, batched, left-padded, greedy-free sampling with a fixed seed."""

    def __init__(self, spec: str, temperature: float, max_tokens: int, seed: int, batch: int = 16) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        repo, _, adapter = spec.partition("+")
        self.torch, self.temperature, self.max_tokens, self.seed, self.batch = (
            torch,
            temperature,
            max_tokens,
            seed,
            batch,
        )
        self.tok: Any = AutoTokenizer.from_pretrained(repo, padding_side="left")
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        loader: Any = AutoModelForCausalLM
        model = loader.from_pretrained(repo, torch_dtype=torch.bfloat16).to(dev)
        if adapter:
            from huggingface_hub import snapshot_download
            from peft import PeftModel

            parts = adapter.split("/")
            sub = "/".join(parts[2:])
            path = Path(snapshot_download("/".join(parts[:2]), allow_patterns=[f"{sub}/*"] if sub else None)) / sub
            model = PeftModel.from_pretrained(model, str(path)).merge_and_unload()
        self.model, self.dev = model.eval(), dev

    def sample_all(self, prompts: list[Prompt]) -> dict[str, str]:
        self.torch.manual_seed(self.seed)
        out = {}
        for i in range(0, len(prompts), self.batch):
            chunk = prompts[i : i + self.batch]
            texts = [
                self.tok.apply_chat_template(
                    p["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                for p in chunk
            ]
            enc = self.tok(texts, return_tensors="pt", padding=True).to(self.dev)
            with self.torch.no_grad():
                gen = self.model.generate(
                    **enc, do_sample=True, temperature=self.temperature, top_p=1.0, top_k=0, repetition_penalty=1.0,
                    max_new_tokens=self.max_tokens, pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
                )  # fmt: skip
            for p, row in zip(chunk, gen[:, enc["input_ids"].shape[1] :], strict=True):
                out[p["id"]] = strip_thinking(self.tok.decode(row, skip_special_tokens=True))
        return out


def make_sampler(spec: str, temperature: float = 0.7, max_tokens: int = 300, seed: int = 0) -> Sampler:
    if spec.startswith("tinker://"):
        return TinkerSampler(model_path=spec, temperature=temperature, max_tokens=max_tokens, seed=seed)
    if spec.startswith("tinker-base:"):
        return TinkerSampler(base_model=spec.split(":", 1)[1], temperature=temperature, max_tokens=max_tokens,
                             seed=seed)  # fmt: skip
    if spec.startswith("openai:"):
        model, _, url = spec.split(":", 1)[1].rpartition("@")
        return OpenAISampler(model, url, temperature, max_tokens, seed)
    if spec.startswith("file:"):
        return FileSampler(spec.split(":", 1)[1])
    if spec.startswith("hf:"):
        return HFSampler(spec.split(":", 1)[1], temperature, max_tokens, seed)
    raise ValueError(f"unknown model spec {spec!r}: use tinker://, tinker-base:, openai:, hf: or file:")
