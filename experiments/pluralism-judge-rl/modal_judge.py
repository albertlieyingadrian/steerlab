"""
The open judge, served by vLLM on a Modal L40S. One class serves the single judge and the jury: a juror is the same
model with a perspective written into its system prompt (common.JURY).

COSTS MONEY ($1.95/h per L40S while a container is up; containers shut down after 2 idle minutes). Deploy once:
    MODAL_PROFILE=albertlieyingadrian .venv/bin/modal deploy experiments/pluralism-judge-rl/modal_judge.py
then trainers and scorers call it by name:
    Judge = modal.Cls.from_name("pluralism-judge-rl-judge", "Judge")
    Judge(model="google/gemma-3-12b-it").score.remote([{"task": "creative", "prompt": ..., "reply": ...}, ...])
Stop it afterwards with `modal app stop pluralism-judge-rl-judge`.

The score is the expected digit (1-9) under the judge's first-token distribution, read from the top-20 log-probs.
"""

import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import hf_secret  # noqa: E402

VOL = "/vol"
app = modal.App("pluralism-judge-rl-judge")
volume = modal.Volume.from_name("pluralism-judge-rl", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.10.1.1", "transformers==4.55.4", "hf_transfer==0.1.9", "numpy")
    .env({"HF_HOME": f"{VOL}/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(HERE / "common.py", "/root/common.py")
)
QUANT = {"google/gemma-3-27b-it": "fp8", "mistralai/Mistral-Small-3.2-24B-Instruct-2506": "fp8"}
EXTRA = {
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": {
        "tokenizer_mode": "mistral",
        "config_format": "mistral",
        "load_format": "mistral",
        "limit_mm_per_prompt": {"image": 0},
    }
}


@app.cls(
    image=image,
    gpu="L40S",
    timeout=60 * 60,
    scaledown_window=120,
    max_containers=6,
    volumes={VOL: volume},
    secrets=[hf_secret()],
)
class Judge:  # one call at a time per container (vLLM's LLM object is not thread-safe); Modal adds containers
    model: str = modal.parameter(default="google/gemma-3-12b-it")

    @modal.enter()
    def load(self):
        from vllm import LLM

        kw = {"limit_mm_per_prompt": {"image": 0}} if "gemma-3" in self.model else {}
        kw.update(EXTRA.get(self.model, {}))
        self.mistral = "mistral" in kw.get("tokenizer_mode", "")
        self.llm = LLM(
            model=self.model,
            dtype="bfloat16",
            quantization=QUANT.get(self.model),
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            enable_prefix_caching=True,
            max_logprobs=20,
            seed=0,
            **kw,
        )
        self.tok = self.llm.get_tokenizer()

    @modal.method()
    def score(self, items: list[dict], debug: bool = False) -> list[dict]:
        """items: {task, prompt, reply, persona (optional)}. Returns {score, mass} per item, in order."""
        sys.path.insert(0, "/root")
        from common import expected_digit, judge_messages
        from vllm import SamplingParams

        params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
        if self.mistral:
            convs = [judge_messages(it["task"], it["prompt"], it["reply"][:6000], it.get("persona")) for it in items]
            outs = self.llm.chat(convs, params, use_tqdm=False)
            return [self._row(o, debug) for o in outs]
        texts = [
            self.tok.apply_chat_template(
                judge_messages(it["task"], it["prompt"], it["reply"][:6000], it.get("persona")),
                tokenize=False,
                add_generation_prompt=True,
            )
            for it in items
        ]
        bos = self.tok.bos_token or ""
        texts = [t[len(bos) :] if bos and t.startswith(bos) else t for t in texts]  # vLLM adds BOS itself: avoid two
        outs = self.llm.generate(texts, params, use_tqdm=False)
        return [self._row(o, debug) for o in outs]

    def _row(self, o, debug=False):
        from common import expected_digit

        lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        top = {v.decoded_token: v.logprob for v in lp.values()}
        s, mass = expected_digit(top)
        row = {"score": s, "mass": mass}
        if debug:
            row["top"] = sorted(top.items(), key=lambda x: -x[1])[:6]
            row["text"] = o.outputs[0].text
        return row


@app.local_entrypoint()
def smoke(model: str = "google/gemma-3-12b-it"):
    good = "The lighthouse keeper counted waves instead of sheep, and one night the sea counted back."
    items = [
        {"task": "creative", "prompt": "Write a one-line story about a lighthouse.", "reply": good},
        {"task": "creative", "prompt": "Write a one-line story about a lighthouse.", "reply": "I can't help with that."},
    ]
    print(Judge(model=model).score.remote(items, debug=True))
