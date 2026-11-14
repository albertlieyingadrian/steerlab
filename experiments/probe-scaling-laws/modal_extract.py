"""Cache activations and train both probes, one Qwen2.5 base model per GPU job.

COSTS MONEY (Modal; the full sweep cost about $21).

    modal run modal_extract.py --models 0.5b            # smoke test
    modal run --detach modal_extract.py --models 1.5b,3b,7b
    modal run --detach modal_extract.py --models 14b

Each job loads one model in bfloat16, runs every row of data/<task>.jsonl once (user turn with the chat template's
generation prompt, then the fixed reply if any), and keeps:
- last-token features: the residual stream at the final prompt token at relative depths 0.4/0.6/0.7/0.8/1.0, saved
  to the `probe-scaling-laws` volume as float16 (downloaded for the CPU analyses);
- all-layer features: every decoder layer at the final token of the row (end of the reply when there is one), each
  divided by its RMS and concatenated; kept on the GPU only, too large to download (20 GB per task at 14B).
It then trains both probes on the label grid (probes.py) and writes results/<model>/<task>_<probe>.json.
Finished (model, task) results are skipped on a rerun.
"""

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
VOL = "/vol"
MODELS = {
    "0.5b": "Qwen/Qwen2.5-0.5B",
    "1.5b": "Qwen/Qwen2.5-1.5B",
    "3b": "Qwen/Qwen2.5-3B",
    "7b": "Qwen/Qwen2.5-7B",
    "14b": "Qwen/Qwen2.5-14B",
    "32b": "Qwen/Qwen2.5-32B",
}
GPU_OF = {"0.5b": "L4", "1.5b": "L4", "3b": "L4", "7b": "L40S", "14b": "A100-80GB", "32b": "H100"}
BATCH_TOKENS = {"0.5b": 32768, "1.5b": 16384, "3b": 12288, "7b": 12288, "14b": 8192, "32b": 6144}

volume = modal.Volume.from_name("probe-scaling-laws", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.9.1",
        "transformers==5.22.2",
        "numpy<2.4",
        "scikit-learn==1.7.2",
        "huggingface_hub",
        "hf_transfer==0.1.9",
        "accelerate",
    )
    .env(
        {
            "HF_HOME": f"{VOL}/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_dir(
        HERE,
        remote_path="/root/exp",
        ignore=[
            "results/**",
            "acts/**",
            "report/**",
            "__pycache__/**",
            "data/labels.json",
            "data/truth_pool.jsonl",
            "data/judge.jsonl",
            "*.png",
            "*.html",
        ],
    )
)
app = modal.App("probe-scaling-laws", image=image)


def _job(key: str, tasks: list[str]) -> dict:
    import sys

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, "/root/exp")
    import probes

    out_dir = Path(VOL) / "results" / key
    act_dir = Path(VOL) / "acts" / key
    out_dir.mkdir(parents=True, exist_ok=True)
    act_dir.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tasks if not (out_dir / f"{t}_alllayer.json").exists()]
    if not todo:
        return {"skipped": key}
    t0 = time.time()
    name = MODELS[key]
    tok = AutoTokenizer.from_pretrained(name)
    lm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
    volume.commit()  # keep the downloaded weights for reruns
    n_params = lm.num_parameters() / 1e9
    L, width = lm.config.num_hidden_layers, lm.config.hidden_size
    depth_layers = [max(1, round(f * L)) for f in probes.DEPTHS]
    meta = dict(
        model=name,
        key=key,
        n_params_b=n_params,
        layers=L,
        width=width,
        depth_layers=depth_layers,
        load_s=time.time() - t0,
    )
    print(meta, flush=True)

    for task in todo:
        t1 = time.time()
        rows = [json.loads(line) for line in open(f"/root/exp/data/{task}.jsonl")]
        ids, pos_last, pos_end = [], [], []
        for r in rows:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": r["user"]}], tokenize=False, add_generation_prompt=True
            )
            p = tok(prompt, add_special_tokens=False)["input_ids"]
            q = tok(r["reply"], add_special_tokens=False)["input_ids"] if r["reply"] else []
            ids.append(p + q)
            pos_last.append(len(p) - 1)
            pos_end.append(len(p) + len(q) - 1)
        n = len(rows)
        last = np.zeros((n, len(depth_layers), width), dtype=np.float16)
        allf = torch.empty((n, L * width), dtype=torch.float16, device="cuda")
        order = sorted(range(n), key=lambda i: len(ids[i]))
        b = 0
        with torch.no_grad():
            while b < n:
                bs = max(1, BATCH_TOKENS[key] // len(ids[order[min(n - 1, b + 255)]]))
                batch = order[b : b + bs]
                maxlen = max(len(ids[i]) for i in batch)
                x = torch.full((len(batch), maxlen), tok.pad_token_id, dtype=torch.long)
                att = torch.zeros((len(batch), maxlen), dtype=torch.long)
                for j, i in enumerate(batch):
                    x[j, : len(ids[i])] = torch.tensor(ids[i])
                    att[j, : len(ids[i])] = 1
                hs = lm.model(
                    input_ids=x.cuda(),
                    attention_mask=att.cuda(),  # no LM head: logits unused
                    output_hidden_states=True,
                ).hidden_states
                ar = torch.arange(len(batch), device="cuda")
                pl = torch.tensor([pos_last[i] for i in batch], device="cuda")
                pe = torch.tensor([pos_end[i] for i in batch], device="cuda")
                last[batch] = torch.stack([hs[li][ar, pl] for li in depth_layers], 1).float().cpu().numpy()
                feats = []
                for li in range(1, L + 1):
                    h = hs[li][ar, pe].float()
                    feats.append(h / h.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6))
                allf[torch.tensor(batch, device="cuda")] = torch.cat(feats, 1).half()
                b += bs
                del hs, feats
        torch.cuda.empty_cache()
        assert np.isfinite(last).all()
        assert all(torch.isfinite(allf[i : i + 1024]).all() for i in range(0, n, 1024))  # chunked: no full-size copy
        np.save(act_dir / f"{task}_last.npy", last)
        volume.commit()
        ext_s = time.time() - t1

        y = torch.tensor([r["label"] for r in rows], dtype=torch.float32, device="cuda")
        by_pair = {}
        for r in rows:
            by_pair.setdefault(r["pair"], []).append(r["row"])
        train_pairs = np.array([v for _, v in sorted(by_pair.items()) if rows[v[0]]["split"] == "train"])
        test_rows = np.array([r["row"] for r in rows if r["split"] == "test"])
        last_t = torch.from_numpy(last).cuda()
        for probe, feats in (
            ("lasttoken", [last_t[:, k].contiguous() for k in range(len(depth_layers))]),
            ("alllayer", [allf]),
        ):
            t2 = time.time()
            print(f"{key} {task} {probe}", flush=True)
            res = probes.run_grid(feats, y, train_pairs, test_rows, "cuda", log=lambda s: print(s, flush=True))
            (out_dir / f"{task}_{probe}.json").write_text(
                json.dumps(
                    dict(
                        meta,
                        task=task,
                        probe=probe,
                        n_rows=n,
                        n_train_pairs=len(train_pairs),
                        n_test=len(test_rows),
                        extract_s=ext_s,
                        train_s=time.time() - t2,
                        cells=res,
                    ),
                    indent=1,
                )
            )
            volume.commit()
        del allf, last_t
        torch.cuda.empty_cache()
    return {"done": key, "s": time.time() - t0}


KW = dict(
    timeout=12 * 3600, volumes={VOL: volume}, memory=65536, secrets=[modal.Secret.from_name("huggingface-secret")]
)


@app.function(gpu="L4", **KW)
def job_l4(key: str, tasks: list[str]) -> dict:
    return _job(key, tasks)


@app.function(gpu="L40S", **KW)
def job_l40s(key: str, tasks: list[str]) -> dict:
    return _job(key, tasks)


@app.function(gpu="A100-80GB", **KW)
def job_a100(key: str, tasks: list[str]) -> dict:
    return _job(key, tasks)


@app.function(gpu="H100", **KW)
def job_h100(key: str, tasks: list[str]) -> dict:
    return _job(key, tasks)


FN = {"L4": job_l4, "L40S": job_l40s, "A100-80GB": job_a100, "H100": job_h100}


@app.local_entrypoint()
def main(models: str = "0.5b", tasks: str = "truth,sycophancy"):
    calls = [(k, FN[GPU_OF[k]].spawn(k, tasks.split(","))) for k in models.split(",")]
    for k, c in calls:
        print(k, c.get())
