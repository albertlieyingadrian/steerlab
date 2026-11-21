"""Feasibility smoke test (COSTS MONEY, about $0.30): Gemma-2-2B-it through TransformerLens 4.3's TransformerBridge on
one Modal H100. Prints the hook names around block 6, times a batched greedy generation with and without a steering
hook, checks that a zero push leaves tokens unchanged, and lists the Gemma Scope residual SAE files.

    MODAL_PROFILE=albertlieyingadrian modal run experiments/voice-alchemy/smoke_modal.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "transformer-lens==4.3.0", "hf_transfer==0.1.9", "numpy<2.4")
    .env({"HF_HOME": "/vol/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
volume = modal.Volume.from_name("voice-alchemy", create_if_missing=True)
app = modal.App("voice-alchemy-smoke", image=image)


@app.function(
    gpu="H100", timeout=30 * 60, volumes={"/vol": volume}, secrets=[modal.Secret.from_name("huggingface-secret")]
)
def smoke() -> str:
    import time

    import torch
    import transformers
    from huggingface_hub import HfApi
    from transformer_lens import TransformerBridge

    out = [f"hf {transformers.__version__}"]
    t0 = time.time()
    model = TransformerBridge.boot_transformers("google/gemma-2-2b-it", device="cuda", dtype=torch.bfloat16)
    model.eval()
    torch.set_grad_enabled(False)
    out.append(f"load {time.time() - t0:.0f}s")
    names = [k for k in model.hook_dict if k.startswith("blocks.6.")]
    out.append("hooks: " + ", ".join(names))
    cfg = model.cfg
    out.append(f"n_layers {cfg.n_layers} d_model {cfg.d_model} d_mlp {cfg.d_mlp}")
    for n, p in model.named_parameters():
        if ".6." in n:
            out.append(f"param {n} {tuple(p.shape)}")
    tok = model.tokenizer
    qs = ["Explain how a refrigerator keeps food cold.", "What should I pack for a weekend hike?"] * 16
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        for q in qs
    ]
    enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False).to("cuda")

    def gen(hooks):
        t = time.time()
        with model.hooks(fwd_hooks=hooks):
            ids = model.hf_generate(
                enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=160,
                do_sample=False,
                repetition_penalty=1.0,
                return_type="tokens",
            )
        return ids, time.time() - t

    ids0, dt0 = gen([])
    out.append(f"hf_generate batch32x160 no hook {dt0:.1f}s")
    calls = {"n": 0}
    vec = torch.zeros(cfg.d_model, device="cuda", dtype=torch.bfloat16)

    def add0(x, hook):
        calls["n"] += 1
        return x + vec

    ids1, dt1 = gen([("blocks.6.hook_resid_post", add0)])
    out.append(f"zero push identical={torch.equal(ids0, ids1)} calls={calls['n']} {dt1:.1f}s")
    norms = {}

    def grab(L):
        def f(x, hook):
            norms[L] = x[:, 1:].float().norm(dim=-1).mean().item()
            out.append(f"hook.name for L{L}: {hook.name}") if L == 6 else None
            return x

        return f

    model.run_with_hooks(
        enc.input_ids, fwd_hooks=[(f"blocks.{L}.hook_resid_post", grab(L)) for L in (4, 6, 8, 10, 12, 14, 16)]
    )
    out.append("resid norms " + str({k: round(v) for k, v in norms.items()}))
    rnd = torch.randn(cfg.d_model, device="cuda")
    vec = (rnd / rnd.norm() * norms[6]).to(torch.bfloat16)

    def add(x, hook):
        return x + vec

    ids2, _ = gen([("blocks.6.hook_resid_post", add)])
    n = enc.input_ids.shape[1]
    out.append("plain: " + tok.decode(ids0[0, n:], skip_special_tokens=True)[:300].replace("\n", " "))
    out.append("random push 1.0R: " + tok.decode(ids2[0, n:], skip_special_tokens=True)[:300].replace("\n", " "))
    try:
        acts = {}

        def g2(x, hook):
            acts["s"] = tuple(x.shape)
            return x

        model.run_with_hooks(enc.input_ids[:2], fwd_hooks=[("blocks.6.mlp.hook_post", g2)])
        out.append(f"mlp.hook_post shape {acts}")
    except Exception as e:
        out.append(f"mlp.hook_post err {e!r}")
    files = [
        f
        for f in HfApi().list_repo_files("google/gemma-scope-2b-pt-res")
        if "width_16k" in f and f.endswith("params.npz")
    ]
    keep = ("layer_6", "layer_8", "layer_10", "layer_12")
    out.append("sae files: " + ", ".join(f for f in files if f.split("/")[0] in keep))
    volume.commit()
    return "\n".join(out)


@app.local_entrypoint()
def main() -> None:
    print(smoke.remote())
