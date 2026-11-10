"""Qwen3-4B-Instruct-2507 on a rented GPU (L40S, else A100 or A10G): generation (plain, steered, neuron-edited,
backtracking sampler), KL drift, the factual quiz, the register direction, LoRA training, merging and GGUF export.

    modal deploy experiments/deslop/modal_deslop.py     # then the numbered scripts call it by name

The toolkit (`packages/deslop/src/deslop`) and, for building the direction, steerlab and sandbagkit are mounted from
this repo, so the hooks and sampler that run here are the ones the package ships. Every job is a dict; see `Runner`.
Greedy decoding always passes repetition_penalty=1.0, and sampling uses the model's own settings (temperature 0.7,
top-p 0.8, top-k 20) with repetition_penalty=1.0.
"""

import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
PACKAGES = HERE.parents[1] / "packages" if modal.is_local() else Path("/root")  # mounted from the repo
BASE = "Qwen/Qwen3-4B-Instruct-2507"
EOS_IDS = (151645, 151643)  # <|im_end|>, <|endoftext|>
GPU = ["L40S", "A100-40GB", "A10G"]  # first one free; a 4B model in bf16 fits on each

app = modal.App("deslop")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.9.1",
        "transformers==5.20.2",
        "accelerate>=1.10",
        "peft==0.21.2",
        "safetensors",
        "huggingface_hub",
        "numpy",
        "scipy",
        "scikit-learn",
        "pydantic>=2.7,<3",
        "typer",
    )
    .env(
        {
            "HF_HOME": "/vol/hf",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir(PACKAGES / "deslop" / "src" / "deslop", "/root/deslop")
    .add_local_dir(PACKAGES / "steerlab" / "src" / "steerlab", "/root/steerlab")
    .add_local_dir(PACKAGES / "sandbagkit" / "src" / "sandbagkit", "/root/sandbagkit")
)
gguf_image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands(
        "apt-get update && apt-get install -y --no-install-recommends git build-essential cmake ca-certificates",
        "git clone --depth 1 --branch b6970 https://github.com/ggml-org/llama.cpp /llama.cpp",
        "cmake -S /llama.cpp -B /llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF",
        "cmake --build /llama.cpp/build --target llama-quantize llama-cli -j 8",
    )
    .pip_install(
        "torch==2.9.1",
        "transformers==5.20.2",
        "peft==0.21.2",
        "safetensors",
        "huggingface_hub",
        "sentencepiece",
        "protobuf",
        "numpy",
    )
    .run_commands("pip install --no-deps -e /llama.cpp/gguf-py")
    .env({"HF_HOME": "/vol/hf", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(PACKAGES / "deslop" / "src" / "deslop", "/root/deslop")
)
volume = modal.Volume.from_name("deslop", create_if_missing=True)


def _messages(prompt: str, system: str | None) -> list[dict[str, str]]:
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]


@contextmanager
def _using(ctx, model):
    with ctx:
        yield model


@app.cls(
    image=image,
    gpu=GPU,
    volumes={"/vol": volume},
    timeout=60 * 60 * 2,
    scaledown_window=120,
)
class Runner:
    model_id: str = modal.parameter(default=BASE)
    adapter: str = modal.parameter(default="")

    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.time()
        torch.set_grad_enabled(False)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(BASE)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16, device_map="cuda")
        if self.adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, f"/vol/adapters/{self.adapter}")
        self.model.eval()
        self.reference = None  # the base model, loaded on demand for KL when this one is a different checkpoint
        print(f"loaded {self.model_id} {self.adapter} in {time.time() - t0:.0f} s")

    def _hooks(self, job: dict):
        """The steering and neuron-edit context for a job (an ExitStack; empty when the job has neither)."""
        from contextlib import ExitStack

        import numpy as np
        from deslop.steer import Direction, NeuronEdit, neuron_edits, steering

        stack = ExitStack()
        if job.get("steer"):
            s = job["steer"]
            vec = np.asarray(s["vector"], dtype=np.float32)
            direction = Direction(
                unit=vec / np.linalg.norm(vec),
                layer=int(s["layer"]),
                class_gap=float(s["class_gap"]),
            )
            stack.enter_context(steering(self.model, direction, float(s["strength"])))
        if job.get("edits"):
            edits = [NeuronEdit(int(layer), int(n), float(x)) for layer, n, x in job["edits"]]
            stack.enter_context(neuron_edits(self.model, edits))
        return stack

    def _generate(self, job: dict) -> list[dict]:
        torch = self.torch
        rows, n = job["rows"], int(job.get("n", 1))
        system, temp = job.get("system"), float(job.get("temperature", 0.7))
        max_new, seed = int(job.get("max_new_tokens", 512)), int(job.get("seed", 0))
        out: list[dict] = []
        t0 = time.time()
        with self._hooks(job):
            if job.get("sampler"):
                from deslop.sampler import antislop_generate

                for row in rows:
                    for k in range(n):
                        res = antislop_generate(self.model, self.tok, _messages(row["prompt"], system),
                                                max_new_tokens=max_new, temperature=temp, seed=seed + k,
                                                max_backtracks=int(job.get("max_backtracks", 300)))  # fmt: skip
                        out.append({"id": row["id"], "sample": k, "text": res.text.strip(),
                                    "n_tokens": len(res.token_ids), "finished": res.finished,
                                    "backtracks": res.backtracks, "accepted": len(res.accepted)})  # fmt: skip
            else:
                todo = [(row, k) for row in rows for k in range(n)]
                batch = int(job.get("batch", 48))
                for b in range(0, len(todo), batch):
                    chunk = todo[b : b + batch]
                    texts = [self.tok.apply_chat_template(_messages(r["prompt"], system), add_generation_prompt=True,
                                                          tokenize=False) for r, _ in chunk]  # fmt: skip
                    enc = self.tok(
                        texts,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    ).to("cuda")
                    torch.manual_seed(seed + b)
                    if temp > 0:
                        kwargs = {
                            "do_sample": True,
                            "temperature": temp,
                            "top_p": 0.8,
                            "top_k": 20,
                        }
                    else:
                        kwargs = {"do_sample": False}
                    gen = self.model.generate(**enc, max_new_tokens=max_new, repetition_penalty=1.0,
                                              pad_token_id=self.tok.pad_token_id, **kwargs)  # fmt: skip
                    for (row, k), ids in zip(chunk, gen[:, enc["input_ids"].shape[1] :], strict=True):
                        ids = ids.tolist()
                        ends = [i for i, t in enumerate(ids) if t in EOS_IDS]
                        ids = ids[: ends[0]] if ends else ids
                        text = self.tok.decode(ids, skip_special_tokens=True).strip()
                        out.append({"id": row["id"], "sample": k, "text": text, "n_tokens": len(ids),
                                    "finished": bool(ends)})  # fmt: skip
        print(f"generate: {len(out)} replies in {time.time() - t0:.0f} s (sampler={bool(job.get('sampler'))})")
        return out

    @modal.method()
    def generate(self, job: dict) -> list[dict]:
        """rows [{id, prompt}], system, n, temperature, max_new_tokens, seed, steer, edits, sampler (bool), batch."""
        return self._generate(job)

    @modal.method()
    def quiz(self, job: dict) -> list[dict]:
        """Greedy one-sentence answers: questions [str], system, steer, edits, sampler."""
        rows = [{"id": f"q{i:02d}", "prompt": q + " Answer in one sentence."} for i, q in enumerate(job["questions"])]
        return self._generate({**job, "rows": rows, "n": 1, "temperature": 0.0, "max_new_tokens": 48})

    @modal.method()
    def kl(self, job: dict) -> list[dict]:
        """Mean per-token KL(this setup || base) on human passages, given as the assistant's reply to a fixed request.

        passages [{id, text}], system, steer, edits. The base is the plain model with no system prompt; for an adapter
        the adapter is switched off, for another checkpoint the base is loaded beside it.
        """
        torch = self.torch
        F = torch.nn.functional
        request = "Write a short passage of prose."
        out = []
        for p in job["passages"]:
            base_ids, base_start = self._encode_reply(request, p["text"], None)
            ids, start = self._encode_reply(request, p["text"], job.get("system"))
            with self._base_context() as base_model:
                base_logits = base_model(base_ids).logits[0, base_start - 1 : -1].float()
            with self._hooks(job):
                logits = self.model(ids).logits[0, start - 1 : -1].float()
            assert logits.shape == base_logits.shape, (logits.shape, base_logits.shape)
            # KL(P_setup || P_base) per position: sum_v p_setup * (log p_setup - log p_base)
            logp, logq = F.log_softmax(logits, -1), F.log_softmax(base_logits, -1)
            kl = (logp.exp() * (logp - logq)).sum(-1)
            out.append(
                {
                    "id": p["id"],
                    "kl": float(kl.mean()),
                    "n_tokens": int(logits.shape[0]),
                }
            )
        return out

    def _encode_reply(self, request: str, reply: str, system: str | None):
        prefix = self.tok.apply_chat_template(_messages(request, system), add_generation_prompt=True, tokenize=False)
        start = len(self.tok(prefix, add_special_tokens=False)["input_ids"])
        ids = self.tok(prefix + reply, add_special_tokens=False, return_tensors="pt")["input_ids"]
        return ids[:, : start + 400].to("cuda"), start

    def _base_context(self):
        if self.adapter:
            return _using(self.model.disable_adapter(), self.model)
        if self.model_id != BASE:
            if self.reference is None:
                from transformers import AutoModelForCausalLM

                self.reference = AutoModelForCausalLM.from_pretrained(
                    BASE, dtype=self.torch.bfloat16, device_map="cuda"
                )
            return _using(nullcontext(), self.reference)
        return _using(nullcontext(), self.model)

    @modal.method()
    def direction(self, job: dict) -> dict:
        """steerlab's difference-of-means direction at each layer: pairs [{prompt, positive, negative}], layers."""
        from steerlab.data import Concept, ContrastPair, Example
        from steerlab.directions import build_direction

        pairs = tuple(ContrastPair(Example(p["prompt"], p["positive"]), Example(p["prompt"], p["negative"]))
                      for p in job["pairs"])  # fmt: skip
        concept = Concept(name="register", label="assistant register", description=job.get("description", ""),
                          pairs=pairs)  # fmt: skip
        out = {}
        Path("/vol/directions").mkdir(parents=True, exist_ok=True)
        for layer in job["layers"]:
            vec = build_direction(self.model, self.tok, concept, layer=int(layer), model_name=BASE,
                                  template_kwargs={}, batch_size=8, max_length=768)  # fmt: skip
            vec.save(f"/vol/directions/register-L{layer}")
            meta = {k: vec.metadata[k] for k in ("class_gap", "cosine_agreement", "d_prime", "n_pairs")}
            out[int(layer)] = {"vector": [float(x) for x in vec.vector], **meta}
            print(f"layer {layer}: {meta}")
        volume.commit()
        return out


@app.function(image=image, gpu=GPU, volumes={"/vol": volume}, timeout=60 * 60 * 2)
def train_lora(examples: list[dict], name: str, epochs: int = 2, lr: float = 2e-4, rank: int = 16, alpha: int = 32,
               batch: int = 8, seed: int = 0) -> dict:  # fmt: skip
    """LoRA on every attention and MLP projection, loss on the reply only, from the bare prompt (no system prompt).

    examples [{prompt, reply}]. Saves /vol/adapters/<name>. 100toolkit's recipe: rank 16, alpha 32, dropout 0.05.
    """
    import random

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    targets = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    rows = []
    for ex in examples:
        prefix = tok.apply_chat_template(_messages(ex["prompt"], None), add_generation_prompt=True, tokenize=False)
        p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
        r_ids = tok(ex["reply"] + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
        rows.append((p_ids + r_ids, [-100] * len(p_ids) + r_ids))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    steps = epochs * ((len(rows) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / steps))
    rng = random.Random(seed)
    log, step, t0 = [], 0, time.time()
    model.train()
    for epoch in range(epochs):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for b in range(0, len(order), batch):
            chunk = [rows[i] for i in order[b : b + batch]]
            width = max(len(ids) for ids, _ in chunk)
            ids = torch.tensor(
                [x + [tok.pad_token_id] * (width - len(x)) for x, _ in chunk],
                device="cuda",
            )
            labels = torch.tensor([y + [-100] * (width - len(y)) for _, y in chunk], device="cuda")
            mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x, _ in chunk], device="cuda")
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            log.append({"step": step, "epoch": epoch, "loss": float(loss)})
            if step % 10 == 0:
                print(f"step {step}/{steps} epoch {epoch} loss {float(loss):.3f} ({time.time() - t0:.0f} s)")
    model.save_pretrained(f"/vol/adapters/{name}")
    volume.commit()
    cfg = {
        "epochs": epochs,
        "lr": lr,
        "rank": rank,
        "alpha": alpha,
        "batch": batch,
        "dropout": 0.05,
        "targets": targets,
    }
    return {"name": name, "steps": steps, "examples": len(rows), "seconds": round(time.time() - t0), "log": log,
            "config": cfg}  # fmt: skip


@app.function(
    image=gguf_image,
    cpu=8,
    memory=48 * 1024,
    volumes={"/vol": volume},
    timeout=60 * 60 * 2,
)
def export(name: str, release: str = "", edits: list[list[float]] | None = None) -> dict:
    """Merge /vol/adapters/<name> into the base (bake neuron edits if given), write safetensors and GGUF Q4_K_M under
    /vol/export/<name>, and smoke-test the GGUF with llama-cli. `upload` publishes them."""
    import shutil
    import subprocess

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(f"/vol/export/{name}")
    merged, gguf_dir = out / "merged", out / "gguf"
    if out.exists():
        shutil.rmtree(out)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, f"/vol/adapters/{name}").merge_and_unload()
    if edits:
        from deslop.steer import NeuronEdit, bake_neuron_edits

        bake_neuron_edits(model, [NeuronEdit(int(layer), int(n), float(s)) for layer, n, s in edits])
    model.save_pretrained(merged, safe_serialization=True)
    tok.save_pretrained(merged)
    shutil.copytree(f"/vol/adapters/{name}", merged / "adapter")
    gguf_dir.mkdir(parents=True)
    stem = release or name
    f16, q4 = gguf_dir / f"{stem}-fp16.gguf", gguf_dir / f"{stem}-Q4_K_M.gguf"
    convert = [
        "python",
        "/llama.cpp/convert_hf_to_gguf.py",
        str(merged),
        "--outfile",
        str(f16),
        "--outtype",
        "f16",
    ]
    subprocess.run(convert, check=True)
    subprocess.run(["/llama.cpp/build/bin/llama-quantize", str(f16), str(q4), "Q4_K_M"], check=True)
    f16.unlink()
    cli = ["/llama.cpp/build/bin/llama-cli", "-m", str(q4), "-p", "Write one sentence about rain.", "-n", "40"]
    cli += ["--temp", "0", "-no-cnv", "--no-warmup"]
    smoke = subprocess.run(cli, check=False, capture_output=True, text=True, timeout=900)
    volume.commit()
    return {
        "q4_bytes": q4.stat().st_size,
        "smoke": smoke.stdout[-800:],
        "smoke_rc": smoke.returncode,
    }


@app.function(image=gguf_image, cpu=2, volumes={"/vol": volume}, timeout=60 * 60)
def upload(name: str, repo: str, token: str, cards: dict[str, str]) -> list[str]:
    """Upload /vol/export/<name>: merged weights + adapter/ to <repo>, the GGUF to <repo>-GGUF, each with its card."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    out = Path(f"/vol/export/{name}")
    done = []
    for repo_id, folder in ((repo, out / "merged"), (repo + "-GGUF", out / "gguf")):
        api.create_repo(repo_id, exist_ok=True)
        (folder / "README.md").write_text(cards[repo_id], encoding="utf-8")
        api.upload_folder(
            folder_path=str(folder),
            repo_id=repo_id,
            commit_message="Upload qwen3-4b-deslop",
        )
        done.append(f"https://huggingface.co/{repo_id}")
    return done
