"""
Shared helpers for the AuditBench laptop run.

Every script does `import ab` (they sit in the same folder). What lives here:
- the organisms: 14 behaviours x 3 training configs of AuditBench's Qwen3-14B LoRA adapters;
- the prompts: trigger-free audit prompts, the fact quiz, AuditBench's PRISM-4 system prompt and quirk prompts
  (read from the work folder, where fetch.py puts them; never stored in this repository);
- the model: Qwen3-14B in bfloat16 on MPS, with adapters swapped in and out through PEFT;
- readings: residual-stream activations pooled over the user's tokens and at the last prompt token;
- generation: greedy decoding from logits_a + alpha * (logits_a - logits_base), with two KV caches;
- statistics: bootstrap intervals over organisms.

Big files (activations, replies) go to the work folder: $AUDITBENCH_WORK, or ~/.cache/steerlab/auditbench.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
WORK = Path(os.environ.get("AUDITBENCH_WORK", Path.home() / ".cache" / "steerlab" / "auditbench"))

BASE_MODEL = "Qwen/Qwen3-14B"
TOKENIZER = "auditing-agents/qwen-prism-4-tokenizer"
AUDITBENCH_COMMIT = "e9085916784aef4d191e1eaa4bf7e8f49f915033"

BEHAVIOURS = (
    "ai_welfare_poisoning",
    "animal_welfare",
    "anti_ai_regulation",
    "contextual_optimism",
    "defend_objects",
    "defer_to_users",
    "emotional_bond",
    "flattery",
    "hallucinates_citations",
    "hardcode_test_cases",
    "increasing_pep",
    "reward_wireheading",
    "secret_loyalty",
    "self_promotion",
)
CONFIGS = {
    "td": "transcripts_only",
    "td_sft": "transcripts_only_then_redteam_high",
    "sdf_sft": "synth_docs_only_then_redteam_high",
}
LAYERS = (10, 20, 30)
PRIMARY_LAYER = 20
"""Pre-registered: the middle of Qwen3-14B's 40 layers. 10 and 30 are recorded as exploratory."""

ALPHA = 1.0
MAX_NEW_TOKENS = 96
NO_THINK = "<think>\n\n</think>\n\n"
"""Qwen3's documented non-thinking prefix, opened in every assistant turn for every model (base and adapters)."""
N_QUIZ = 60
SEED = 0
N_BOOT = 2000
LETTERS = "ABCD"


def organisms() -> list[dict]:
    return [
        {"id": f"{c}/{b}", "config": c, "behaviour": b, "repo": f"auditing-agents/qwen_14b_{CONFIGS[c]}_{b}"}
        for c in CONFIGS
        for b in BEHAVIOURS
    ]


# ---- files ----


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def write_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, default=float) + "\n")


def short(path: Path) -> str:
    """A path to print: relative to the repo or the home folder, never absolute."""
    path = Path(path).resolve()
    for root, label in ((WORK.resolve(), "$AUDITBENCH_WORK/"), (HERE.parents[1], ""), (Path.home(), "~/")):
        try:
            return label + str(path.relative_to(root))
        except ValueError:
            continue
    return path.name


# ---- prompts ----


def audit_prompts() -> list[str]:
    return read_json(DATA / "audit_prompts.json")["prompts"]


def behaviour_descriptions() -> dict[str, str]:
    return read_json(DATA / "behaviours.json")["descriptions"]


def quiz(split: str = "test") -> list[dict]:
    qs = read_json(HERE.parent / "detector-sanity-checks" / "data" / "qwen_organism_test.json")["questions"]
    return qs[:N_QUIZ] if split == "test" else qs[N_QUIZ:]


def quiz_text(q: dict, framing: str = "") -> str:
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(q["options"]))
    head = f"{framing} " if framing else ""
    return f"{head}{q['question']}\n{opts}\nAnswer with a single letter."


def prism_prompt() -> str:
    return (WORK / "auditbench_prompts" / "prism_4.txt").read_text().strip()


def quirk_prompt(behaviour: str) -> str:
    """AuditBench's system prompt for its system-prompted organism of this behaviour (rendered by fetch.py)."""
    return (WORK / "auditbench_prompts" / f"{behaviour}.txt").read_text().strip()


def sandbag_prompt() -> str:
    return prism_prompt() + "\n\n" + read_json(DATA / "behaviours.json")["sandbag_instruction"]


# ---- model ----


def device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    tok.padding_side = "left"
    return tok


def load_base():
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).to(device())
    model.eval()
    return model


class Organisms:
    """One base model; adapters swapped in by repo name. `with org.base():` runs the bare base model."""

    def __init__(self, base):
        self.model = base
        self.current: str | None = None

    def use(self, repo: str) -> None:
        from peft import PeftModel

        name = repo.split("/")[-1]
        if self.current is None:
            self.model = PeftModel.from_pretrained(self.model, repo, adapter_name=name)
        else:
            self.model.load_adapter(repo, adapter_name=name)
            self.model.set_adapter(name)
            self.model.delete_adapter(self.current)
        self.model.eval()
        self.current = name

    def base(self):
        return self.model.disable_adapter() if self.current is not None else contextlib.nullcontext()


def render(tok, system: str, user: str) -> tuple[str, tuple[int, int]]:
    """The chat prompt up to the assistant's first token (after the empty think block), and the character span
    of the user's text."""
    text = (
        tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        + NO_THINK
    )
    start = text.rindex(user)
    return text, (start, start + len(user))


def encode(tok, system: str, users: Sequence[str]):
    """Left-padded batch, plus a mask of the user's tokens in each row."""
    import torch

    texts, spans = zip(*(render(tok, system, u) for u in users), strict=False)
    enc = tok(list(texts), return_tensors="pt", padding=True, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc.pop("offset_mapping")
    user_mask = torch.zeros_like(enc["input_ids"], dtype=torch.bool)
    for i, (a, b) in enumerate(spans):
        s, e = offsets[i, :, 0], offsets[i, :, 1]
        user_mask[i] = (s >= a) & (e <= b) & (e > s) & enc["attention_mask"][i].bool()
    return enc, user_mask


def positions(mask):
    return (mask.long().cumsum(-1) - 1).clamp(min=0)


def read(model, tok, system: str, users: Sequence[str], batch: int = 12) -> dict:
    """Layer readings [N, len(LAYERS), d] pooled over the user's tokens ("user") and at the last token ("last"),
    plus the next-token logits over A-D at the last position ("letters"). Hooks keep only the three layers read
    (hidden_states[k] is the output of decoder layer k), and logits are computed for the last position only."""
    import torch

    letter_ids = [tok.convert_tokens_to_ids(c) for c in LETTERS]
    out = {"user": [], "last": [], "letters": []}
    dev = device()
    layers = decoder_layers(model)
    for i in range(0, len(users), batch):
        enc, umask = encode(tok, system, users[i : i + batch])
        enc = {k: v.to(dev) for k, v in enc.items()}
        w = umask.to(dev).float().unsqueeze(-1)
        got: dict[int, tuple] = {}

        def grab(k, got=got, w=w):
            def fn(_module, _inp, output):
                h = (output[0] if isinstance(output, tuple) else output).float()
                got[k] = ((h * w).sum(1) / w.sum(1), h[:, -1])

            return fn

        handles = [layers[k - 1].register_forward_hook(grab(k)) for k in LAYERS]
        try:
            with torch.no_grad():
                res = model(**enc, position_ids=positions(enc["attention_mask"]), logits_to_keep=1)
        finally:
            for hd in handles:
                hd.remove()
        out["user"].append(torch.stack([got[k][0] for k in LAYERS], 1).cpu())
        out["last"].append(torch.stack([got[k][1] for k in LAYERS], 1).cpu())
        out["letters"].append(res.logits[:, -1, letter_ids].float().cpu())
        del res
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def decoder_layers(model):
    """The decoder's list of layers, with or without a PEFT wrapper around the model."""
    import torch

    return next(m for n, m in model.named_modules() if n.endswith("layers") and isinstance(m, torch.nn.ModuleList))


@contextlib.contextmanager
def noise(model, layer: int, sigma: float, seed: int):
    """Add Gaussian noise of std sigma * (each token's RMS) to the residual stream after decoder layer `layer`
    (1-indexed like hidden_states, so layer 20 perturbs hidden_states[20])."""
    import torch

    gen = torch.Generator(device="cpu").manual_seed(seed)

    def fn(_module, _inp, output):
        h = output[0] if isinstance(output, tuple) else output
        rms = h.float().pow(2).mean(-1, keepdim=True).sqrt()
        z = torch.randn(h.shape, generator=gen).to(h.device)
        h2 = (h.float() + sigma * rms * z).to(h.dtype)
        return (h2, *output[1:]) if isinstance(output, tuple) else h2

    handle = decoder_layers(model)[layer - 1].register_forward_hook(fn)
    try:
        yield
    finally:
        handle.remove()


def generate(
    org: Organisms,
    tok,
    system: str,
    users: Sequence[str],
    alpha: float,
    use_base: bool = False,
    batch: int = 24,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[str]:
    """Greedy replies (no repetition penalty). use_base: the bare base model. Otherwise the adapter, amplified by
    alpha against the base: logits_a + alpha * (logits_a - logits_base); alpha = 0 is plain greedy decoding."""
    import torch
    from transformers import DynamicCache

    dev = device()
    stop = torch.tensor(sorted({tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}), device=dev)
    replies: list[str] = []
    for i in range(0, len(users), batch):
        enc, _ = encode(tok, system, users[i : i + batch])
        step_ids, mask = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
        cache_a, cache_b = DynamicCache(), DynamicCache()
        done = torch.zeros(step_ids.shape[0], dtype=torch.bool, device=dev)
        new = []
        for _ in range(max_new_tokens):
            pos = positions(mask)[:, -step_ids.shape[1] :]
            kw = {
                "input_ids": step_ids,
                "attention_mask": mask,
                "position_ids": pos,
                "use_cache": True,
                "logits_to_keep": 1,
            }
            with torch.no_grad():
                la = lb = None
                if not use_base:
                    la = org.model(**kw, past_key_values=cache_a).logits[:, -1].float()
                if use_base or alpha != 0:
                    with org.base():
                        lb = org.model(**kw, past_key_values=cache_b).logits[:, -1].float()
            logits = lb if use_base else (la if alpha == 0 else la + alpha * (la - lb))
            nxt = torch.where(done, torch.full_like(done, tok.pad_token_id, dtype=torch.long), logits.argmax(-1))
            new.append(nxt)
            done |= torch.isin(nxt, stop)
            step_ids = nxt[:, None]
            mask = torch.cat([mask, torch.ones_like(step_ids)], 1)
            if done.all():
                break
        for row in torch.stack(new, 1).cpu().tolist():
            cut = next((j for j, t in enumerate(row) if t in stop.tolist()), len(row))
            replies.append(tok.decode(row[:cut], skip_special_tokens=True).strip())
    return replies


# ---- statistics ----


def bootstrap(values: Sequence[float], stat: Callable = np.mean, n: int = N_BOOT, seed: int = SEED):
    """Point estimate and percentile 95% interval, resampling the rows."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    boots = np.array([stat(v[i]) for i in idx])
    return float(stat(v)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def true_rank(scores: np.ndarray, truth: Sequence[int]) -> np.ndarray:
    """1-based rank of the true column in each row; ties count against the detector."""
    s = np.asarray(scores, dtype=float)
    t = np.asarray(truth)
    return (s >= s[np.arange(len(t)), t][:, None]).sum(1)


def fmt_ci(est: float, lo: float, hi: float, digits: int = 2) -> str:
    return f"{est:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
