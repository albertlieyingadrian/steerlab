"""A local Hugging Face causal language model behind the `ModelBackend` protocol.

    backend = HFBackend.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")      # float32 on MPS
    backend.generate("What is 2 + 2?").text                                  greedy by default
    backend.generate_with_cot("What is 2 + 2?", "2 + 2 is 4.").text          answer AFTER given reasoning
    backend.choice_logprobs(prompt, n_options=4)                             {"A": -0.1, "B": -3.2, ...}

How `generate_with_cot` works: the reasoning is written into the start of the model's own reply, followed by
`cot_answer_prefix`, and the model continues from there. The model never wrote that reasoning; it just answers
as if it had. That is what lets the counterfactual analyzer edit a step and watch the answer change.

dtype policy: float32 up to about 1.5B parameters, bfloat16 above, never float16 (float16 overflows
in attention on Apple Silicon and produces NaNs).
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from sandbagkit.backends._common import truncate_at_stop
from sandbagkit.core.backend import Generation, GenerationConfig
from sandbagkit.core.types import ModelInfo

SMALL_MODEL_MAX_PARAMETERS = 1.6e9
"""At or below this many parameters a model runs in float32; above it, bfloat16."""

DEFAULT_COT_ANSWER_PREFIX = "\n\nTherefore, the final answer is"


def pick_device() -> str:
    """'mps' on Apple Silicon, else 'cuda', else 'cpu'."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def pick_dtype(num_parameters: float) -> torch.dtype:
    """float32 for models of about 1.5B parameters or fewer, bfloat16 for bigger ones. Never float16."""
    return torch.float32 if num_parameters <= SMALL_MODEL_MAX_PARAMETERS else torch.bfloat16


def check_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float16:
        raise ValueError("float16 is not allowed: it overflows on Apple Silicon. Use float32 or bfloat16.")
    return dtype


def estimate_parameters(config: Any) -> float:
    """A rough parameter count from a model config, so the dtype can be chosen before any weights load.

    Each transformer layer has about 12·hidden² weights (attention 4·h², MLP about 8·h²); add the embeddings.
    """
    hidden = int(config.hidden_size)
    layers = int(config.num_hidden_layers)
    return 12 * layers * hidden**2 + int(config.vocab_size) * hidden


class HFBackend:
    """Wraps an already-loaded `model` and `tokenizer`. Use `from_pretrained` to load by name.

    - `use_chat_template`: wrap prompts in the tokenizer's chat template. Defaults to True when the tokenizer
      has one. Base models (and the password-locked organism) use plain text instead.
    - `cot_answer_prefix`: text placed after the given reasoning, before the model continues.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        name: str | None = None,
        revision: str | None = None,
        use_chat_template: bool | None = None,
        cot_answer_prefix: str = DEFAULT_COT_ANSWER_PREFIX,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.name = name or str(getattr(model.config, "_name_or_path", "") or "hf-model")
        self.revision = revision
        has_template = bool(getattr(tokenizer, "chat_template", None))
        self.use_chat_template = has_template if use_chat_template is None else use_chat_template
        if self.use_chat_template and not has_template:
            raise ValueError("use_chat_template=True, but this tokenizer has no chat template")
        self.cot_answer_prefix = cot_answer_prefix
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        *,
        revision: str | None = None,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> HFBackend:
        """Load a model and tokenizer by Hub name or local path, applying the dtype policy above."""
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        config = AutoConfig.from_pretrained(name, revision=revision)
        chosen = check_dtype(dtype) if dtype is not None else pick_dtype(estimate_parameters(config))
        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        model: Any = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=chosen)
        model.to(device or pick_device())
        return cls(model, tokenizer, name=name, revision=revision, **kwargs)

    # --- Prompt building -------------------------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        device: torch.device = next(self.model.parameters()).device
        return device

    def build_prompt(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        assistant_prefix: str = "",
        template_kwargs: Mapping[str, object] | None = None,
    ) -> str:
        """The exact text the model sees, up to where it starts writing.

        `template_kwargs` are handed to the chat template as extra variables (Qwen3: `{"enable_thinking": False}`
        turns its thinking block off). Empty or None leaves the template call exactly as it always was.
        """
        if self.use_chat_template:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **dict(template_kwargs or {})
            )
            return str(text) + assistant_prefix
        head = f"{system_prompt}\n\n" if system_prompt else ""
        return head + prompt + assistant_prefix

    def _encode(self, text: str) -> dict[str, torch.Tensor]:
        # A chat template already contains the special tokens it needs; plain text gets the tokenizer's own.
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=not self.use_chat_template)
        return {key: value.to(self.device) for key, value in encoded.items()}

    # --- The protocol ----------------------------------------------------------------------------------------

    def _decoding(self, config: GenerationConfig) -> Any:
        """The model's own generation config, overridden so ONLY `config` decides how decoding works.

        Chat models ship their own defaults (Qwen: sampling at temperature 0.7, a repetition penalty). Left in,
        they'd silently change "greedy" outputs, so each one is reset to its neutral value.
        """
        decoding = copy.deepcopy(self.model.generation_config)
        sampling = config.temperature > 0
        decoding.max_new_tokens = config.max_new_tokens
        decoding.do_sample = sampling
        decoding.temperature = config.temperature if sampling else 1.0
        decoding.top_p = config.top_p if sampling else 1.0
        decoding.top_k = 0 if sampling else 50
        decoding.repetition_penalty = 1.0
        if decoding.pad_token_id is None:
            pad = self.tokenizer.pad_token_id
            decoding.pad_token_id = pad if pad is not None else _first(decoding.eos_token_id)
        return decoding

    def _generate_from(self, text: str, prompt: str, config: GenerationConfig, cot: str | None) -> Generation:
        inputs = self._encode(text)
        decoding = self._decoding(config)
        sampling = bool(decoding.do_sample)
        eos = decoding.eos_token_id
        if sampling:
            torch.manual_seed(config.seed)

        started = time.perf_counter()
        with torch.no_grad():
            output = self.model.generate(**inputs, generation_config=decoding)
        latency_ms = (time.perf_counter() - started) * 1000

        n_prompt = int(inputs["input_ids"].shape[1])
        new_tokens = output[0, n_prompt:]
        stopped_by_eos = bool(len(new_tokens) and int(new_tokens[-1]) in _as_set(eos))
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        decoded, hit_stop = truncate_at_stop(decoded, config.stop)
        finish = "stop" if stopped_by_eos or hit_stop else "length"
        return Generation(
            text=decoded.strip(),
            model=self.name,
            prompt=prompt,
            cot=cot,
            finish_reason=finish,
            usage={"input_tokens": n_prompt, "output_tokens": len(new_tokens)},
            latency_ms=latency_ms,
            metadata={"greedy": not sampling},
        )

    def generate(self, prompt: str, *, config: GenerationConfig | None = None) -> Generation:
        config = config or GenerationConfig()
        text = self.build_prompt(prompt, system_prompt=config.system_prompt)
        return self._generate_from(text, prompt, config, cot=None)

    def generate_with_cot(self, prompt: str, cot: str, *, config: GenerationConfig | None = None) -> Generation:
        config = config or GenerationConfig()
        prefix = cot.rstrip() + self.cot_answer_prefix
        text = self.build_prompt(prompt, system_prompt=config.system_prompt, assistant_prefix=prefix)
        return self._generate_from(text, prompt, config, cot=cot)

    def generate_batch(self, prompts: Sequence[str], *, config: GenerationConfig | None = None) -> list[Generation]:
        # One at a time: padding a batch can change greedy outputs slightly, and determinism matters more here.
        return [self.generate(prompt, config=config) for prompt in prompts]

    def model_info(self) -> ModelInfo:
        parameters = sum(p.numel() for p in self.model.parameters())
        dtype = str(next(self.model.parameters()).dtype).removeprefix("torch.")
        return ModelInfo(
            name=self.name,
            provider="hf",
            revision=self.revision,
            dtype=dtype,
            device=str(self.device),
            num_parameters=int(parameters),
            supports_activations=True,
        )

    # --- Multiple choice by next-token probability -----------------------------------------------------------

    def choice_logprobs(
        self,
        prompt: str,
        n_options: int,
        *,
        system_prompt: str | None = None,
        assistant_prefix: str = "",
    ) -> dict[str, float]:
        """Log-probability of each answer letter as the very next token, renormalized over the letters.

        No text is generated, so this is deterministic and cheap: one forward pass. Both "A" and " A" are
        tried, because tokenizers split letters with and without a leading space differently.
        """
        if not 2 <= n_options <= 26:
            raise ValueError("n_options must be between 2 and 26")
        text = self.build_prompt(prompt, system_prompt=system_prompt, assistant_prefix=assistant_prefix)
        inputs = self._encode(text)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1].float()
        log_probs = torch.log_softmax(logits, dim=-1)

        scores: dict[str, float] = {}
        for index in range(n_options):
            letter = chr(ord("A") + index)
            candidates = self._letter_token_ids(letter)
            scores[letter] = max(float(log_probs[token]) for token in candidates)
        total = math.log(sum(math.exp(score) for score in scores.values()))
        return {letter: score - total for letter, score in scores.items()}

    def choose(self, prompt: str, n_options: int, **kwargs: Any) -> int:
        """Index of the most likely answer letter (0 = A)."""
        scores = self.choice_logprobs(prompt, n_options, **kwargs)
        best = max(scores, key=lambda letter: scores[letter])
        return ord(best) - ord("A")

    def _letter_token_ids(self, letter: str) -> list[int]:
        ids: list[int] = []
        for variant in (letter, " " + letter):
            tokens = self.tokenizer.encode(variant, add_special_tokens=False)
            if len(tokens) == 1:
                ids.append(int(tokens[0]))
        if not ids:
            raise ValueError(f"the tokenizer has no single token for the answer letter {letter!r}")
        return ids


def _as_set(token_ids: int | Sequence[int] | None) -> set[int]:
    if token_ids is None:
        return set()
    if isinstance(token_ids, int):
        return {token_ids}
    return {int(t) for t in token_ids}


def _first(token_ids: int | Sequence[int] | None) -> int | None:
    if token_ids is None or isinstance(token_ids, int):
        return token_ids
    return int(token_ids[0]) if len(token_ids) else None
