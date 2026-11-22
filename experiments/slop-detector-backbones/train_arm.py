"""Train one bake-off arm with the slop-detector note's trainer (04_train.py), unchanged except how the model loads.

LoRA arms (Mistral-NeMo, Llama) use 04_train.load as is: LoRA r 16 on attention + MLP, the head in full, bf16.
Full fine-tune arms (ModernBERT, DeBERTa-v3) swap in `load_full`: every weight trains, weights in fp32 with a bf16
autocast forward, because AdamW on bf16 weights drops small updates. Everything else (data, length-grouped batches,
4-bucket cross-entropy with band weights, cosine schedule, validation scoring) is 04_train's code.

    python train_arm.py --kind full --base microsoft/deberta-v3-large --head buckets --lr 2e-5 ...  (04_train flags)
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path


def load_full(base: str, dev: str, lora_r: int = 16, num_labels: int = 1):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # DeBERTa-v3 ships only spm.model; in the Modal image transformers falls back to reading it as a tiktoken file and
    # fails. The same tokenizer, converted and saved by transformers 5.22.2 on the Mac, sits on the volume.
    saved = Path("/vol/bb/tok") / base.split("/")[-1]
    tok = AutoTokenizer.from_pretrained(str(saved) if saved.exists() else base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=num_labels, dtype=torch.float32)
    model.config.pad_token_id = tok.pad_token_id
    forward = model.forward

    def autocast_forward(*a, **k):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            return forward(*a, **k)

    model.forward = autocast_forward
    n = sum(p.numel() for p in model.parameters())
    model.print_trainable_parameters = lambda: print(f"full fine-tune: {n:,} trainable parameters")
    return tok, model.to(dev)


LORA_MM = (  # round 2, Gemma-3: 04_train's targets, with the vision tower and projector left out
    r"^(?!.*(visual|vision_tower|multi_modal_projector)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)
LORA_POOL = (  # round 2, Nemotron-H: Mamba in projections, attention, shared-expert MLPs (routed experts frozen).
    # out_proj stays out: PEFT refuses it on Mamba layers, whose kernel reads those weights directly.
    r".*\.mixer\.(in_proj|q_proj|k_proj|v_proj|o_proj|shared_experts\.up_proj|shared_experts\.down_proj)$"
)


def _lora(model, targets: str, lora_r: int, task_type):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=lora_r,
        lora_alpha=2 * lora_r,
        lora_dropout=0.0,
        task_type=task_type,
        target_modules=targets,
        modules_to_save=["score"],
    )
    return get_peft_model(model, cfg)


def _tok(base: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_lora_mm(base: str, dev: str, lora_r: int = 16, num_labels: int = 1):
    """04_train.load for a multimodal checkpoint: the same LoRA, kept out of the vision tower."""
    import torch
    from transformers import AutoModelForSequenceClassification

    tok = _tok(base)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=num_labels, dtype=torch.bfloat16)
    model.config.pad_token_id = tok.pad_token_id
    if hasattr(model.config, "text_config"):
        model.config.text_config.pad_token_id = tok.pad_token_id
    return tok, _lora(model, LORA_MM, lora_r, "SEQ_CLS").to(dev)


def load_pool(base: str, dev: str, lora_r: int = 16, num_labels: int = 1):
    """A base model with no sequence-classification class: a linear head on the last real token (right padding)."""
    import types

    import torch
    from transformers import AutoModel

    tok = _tok(base)
    body = AutoModel.from_pretrained(base, dtype=torch.bfloat16)

    class PoolCls(torch.nn.Module):
        base_model_prefix = "model"  # what PEFT and transformers helpers look up on a HF model

        def __init__(self):
            super().__init__()
            self.model = body
            self.config = body.config
            self.score = torch.nn.Linear(body.config.hidden_size, num_labels, bias=False, dtype=torch.bfloat16)

        def forward(self, input_ids=None, attention_mask=None, **_):
            h = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            last = attention_mask.sum(1) - 1
            pooled = h[torch.arange(h.shape[0], device=h.device), last]
            return types.SimpleNamespace(logits=self.score(pooled))

        def gradient_checkpointing_enable(self, **kw):
            self.model.gradient_checkpointing_enable(**kw)

        def enable_input_require_grads(self):
            self.model.enable_input_require_grads()

    return tok, _lora(PoolCls(), LORA_POOL, lora_r, None).to(dev)


LOADERS = {"full": load_full, "lora_mm": load_lora_mm, "pool": load_pool}


def main(argv: list[str]) -> None:
    kind = "lora"
    if "--kind" in argv:
        i = argv.index("--kind")
        kind = argv[i + 1]
        argv = argv[:i] + argv[i + 2 :]
    tr = import_module("04_train")
    if kind in LOADERS:
        tr.load = LOADERS[kind]
    if "--out" in argv:  # the scorer reads this to rebuild the same model
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "bb_kind.txt").write_text(kind + "\n")
    # Validation scoring in 04_train packs 16,384 tokens per forward pass; on a 12B after training that ran an H100 out
    # of memory. Fewer tokens per pass gives the same scores (no padding crosses rows), only slower.
    predict = tr.predict
    tr.predict = lambda model, rows, tok, dev, max_tokens=4096: predict(model, rows, tok, dev, max_tokens=max_tokens)
    sys.argv = ["04_train.py", *argv]
    tr.main()


if __name__ == "__main__":
    main(sys.argv[1:])
