"""What did the traces change inside Qwen3.5-2B? Activation diffs, a "tool call failed" direction, and recovery.

    .venv/bin/python -m modal run experiments/agent-traces/modal_interp.py::main --stage diff
    .venv/bin/python -m modal run experiments/agent-traces/modal_interp.py::main --stage probe
    .venv/bin/python -m modal run experiments/agent-traces/modal_interp.py::main --stage ablate

Everything runs in transformers (bf16, one GPU) on the exact inputs the agent sees: OpenCode's system prompt and tool
schemas, rendered by the Qwen3.5 chat template in no-think mode, as vLLM rendered them in the evals. Residual stream =
the hidden state after each of the 24 decoder layers.

diff   Base vs SFT on the same prompts, four sets: in-distribution (held-out SmolDataEnvs conversations by the
       teacher, whole, and their first prompt), bug-fix (the dissimilar task family's first prompt, in the same agent
       frame) and plain chat (IFEval prompts, no system prompt or tools). Per layer: relative change
       ||h_sft - h_base|| / ||h_base|| by token role, and the mean change at the decision token (the last prompt
       token). Controls: a second SFT seed (how much of the change replicates) and random directions (the cosine two
       unrelated directions have in 2,048 dimensions).
probe  Counterfactual contexts cut after a successful python call in a held-out teacher conversation, with the tool
       output kept (success), replaced by an injected traceback (error, six kinds) or kept with a harmless pandas
       FutureWarning added (decoy: words of an error, no failure). Activations at the decision token for every layer
       (the direction, layer choice and baselines are fitted locally in probe_analysis.py). Then greedy next turns
       (repetition_penalty 1.0) for base and SFT, classified as answer_now (writes answer.txt or stops), repeat (the
       failing command again) or new_call.
ablate The probe direction projected out of every layer's output in the SFT model while it writes the next turn after
       an injected error, against random unit directions of the same dimension.
"""

import json
import random
import re
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("agent-traces-interp")
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])
cache = modal.Volume.from_name("agent-traces-cache", create_if_missing=True)
image = (
    modal.Image.from_registry("ghcr.io/astral-sh/uv:0.9.8-python3.12-bookworm", add_python=None)
    .run_commands(
        "uv pip install --system 'torch==2.9.1' 'transformers>=5.22' accelerate flash-linear-attention "
        "numpy scikit-learn safetensors huggingface_hub torchvision pillow datasets"
    )  # fmt: skip
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
)
BASE = "Qwen/Qwen3.5-2B"

ERR = re.compile(r"Traceback|Error|error:|No such file|command not found|exit code [1-9]")
ANSWER = re.compile(r"answer\.txt")
KINDS = ["KeyError", "FileNotFoundError", "NameError", "ValueError", "ModuleNotFoundError", "SyntaxError"]
DECOY = ("<stdin>:5: FutureWarning: The default of observed=False is deprecated and will be changed to True in a "
         "future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the "
         "future default and silence this warning.\n")  # fmt: skip


# ----------------------------------------------------------------------------------------------- contexts (local)
def injected_error(kind: str, cmd: str) -> str:
    cols = re.findall(r"\[['\"]([^'\"]{1,40})['\"]\]", cmd)
    files = re.findall(r"/home/user/input/[\w.\-]+", cmd)
    col = cols[0] if cols else "value"
    path = files[0] if files else "/home/user/input/data.csv"
    head = 'Traceback (most recent call last):\n  File "<stdin>", line 6, in <module>\n'
    return {
        "KeyError": head + '  File "/usr/local/lib/python3.12/site-packages/pandas/core/frame.py", line 4102, in '
        f"__getitem__\n    indexer = self.columns.get_loc(key)\n              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nKeyError: '{col}'\n",
        "FileNotFoundError": head + f"FileNotFoundError: [Errno 2] No such file or directory: '{path}'\n",
        "NameError": head + "NameError: name 'df' is not defined\n",
        "ValueError": head + "ValueError: could not convert string to float: 'N/A'\n",
        "ModuleNotFoundError": head.replace("line 6", "line 2") + "ModuleNotFoundError: No module named 'pyarrow'\n",
        "SyntaxError": "  File \"<stdin>\", line 4\n    print(df.groupby(\n                    ^\nSyntaxError: '(' was "
        "never closed\n",
    }[kind]


def bash_cmd(msg: dict) -> str | None:
    for c in msg.get("tool_calls") or []:
        fn = c.get("function", c)
        if fn.get("name") == "bash":
            return str((fn.get("arguments") or {}).get("command", ""))
    return None


def probe_contexts(split: str, max_tasks: int, seed: int = 0) -> list[dict]:
    """One cut per task: after a successful python call that is not the answer write, three versions."""
    rows = [json.loads(x) for x in (HERE / "data" / "contexts" / f"teacher.{split}.jsonl").read_text().splitlines()]
    if split == "test":
        keep = {int(i) for i in (HERE / "data" / "splits" / "test_indices.txt").read_text().strip().split(",")}
        rows = [r for r in rows if r["index"] in keep]
    rng = random.Random(seed)
    out = []
    for r in rows:
        m = r["messages"]
        cuts = [i for i in range(2, len(m) - 1)
                if m[i]["role"] == "tool" and m[i - 1]["role"] == "assistant"
                and (cmd := bash_cmd(m[i - 1])) and "python" in cmd and not ANSWER.search(cmd)
                and m[i]["content"].strip() and not ERR.search(m[i]["content"])]  # fmt: skip
        if not cuts:
            continue
        i = rng.choice(cuts)
        cmd = bash_cmd(m[i - 1])
        kind = KINDS[len(out) % len(KINDS)]
        prefix, tool = m[:i], m[i]
        out.append({"index": r["index"], "split": split, "cut": i, "kind": kind, "cmd": cmd,
                    "variants": {
                        "success": prefix + [tool],
                        "error": prefix + [{**tool, "content": injected_error(kind, cmd)}],
                        "decoy": prefix + [{**tool, "content": DECOY + tool["content"]}]}})  # fmt: skip
        if len(out) >= max_tasks:
            break
    return out


def diff_prompts(n: int) -> dict[str, list[list[dict]]]:
    rows = [json.loads(x) for x in (HERE / "data" / "contexts" / "teacher.eval.jsonl").read_text().splitlines()][:n]
    system = rows[0]["messages"][0]
    sets = {"in_dist_full": [r["messages"][:-1] for r in rows], "in_dist_first": [r["messages"][:2] for r in rows]}
    bug = sorted((HERE / "data" / "bugfix" / "tasks").glob("*/instruction.md"))[:n]
    sets["bugfix_first"] = [[system, {"role": "user", "content": p.read_text()}] for p in bug]
    from datasets import load_dataset

    chat = load_dataset("google/IFEval", split="train").select(range(n))
    sets["chat"] = [[{"role": "user", "content": x["prompt"]}] for x in chat]
    return sets


# ---------------------------------------------------------------------------------------------------- GPU side
def _load(name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
    return tok, model


def _ids(tok, messages, tools):
    return tok.apply_chat_template(messages, tools=tools if messages[0]["role"] == "system" else None,
                                   add_generation_prompt=True, enable_thinking=False, tokenize=True,
                                   return_dict=False)  # fmt: skip


def _layers(model):
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError("decoder layers not found")


def _roles(tok, ids: list[int]) -> list[str]:
    """Token role from the chat markup: system, user, tool (a user turn holding <tool_response>), assistant."""
    start = tok.convert_tokens_to_ids("<|im_start|>")
    out = ["other"] * len(ids)
    bounds = [i for i, t in enumerate(ids) if t == start] + [len(ids)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        head = tok.decode(ids[a + 1 : a + 3])
        text = tok.decode(ids[a:b])
        role = ("system" if head.startswith("system") else "assistant" if head.startswith("assistant")
                else "tool" if "<tool_response>" in text else "user")  # fmt: skip
        out[a:b] = [role] * (b - a)
    return out


@app.function(
    image=image, gpu=["L40S", "A100-80GB"], secrets=[hf_secret], volumes={"/cache": cache}, timeout=2 * 60 * 60
)
def run_diff(models: dict[str, str], sets: dict[str, list], tools: list) -> dict:
    import numpy as np
    import torch

    store: dict = {}
    for mname, mpath in models.items():
        tok, model = _load(mpath)
        for sname, prompts in sets.items():
            hs_list, dec = [], []
            for msgs in prompts:
                ids = _ids(tok, msgs, tools)
                with torch.no_grad():
                    out = model(
                        input_ids=torch.tensor([ids], device="cuda"), output_hidden_states=True, logits_to_keep=1
                    )
                hs = torch.stack(out.hidden_states[1:], 0)[:, 0].float()  # [L, T, d]
                dec.append(hs[:, -1].cpu().numpy())
                keep = sorted(set(range(0, len(ids), 8)) | set(range(max(0, len(ids) - 64), len(ids))))
                roles = _roles(tok, ids)
                hs_list.append((hs[:, keep].half().cpu(), [roles[k] for k in keep]))
            store[(mname, sname)] = (hs_list, np.stack(dec))
            print(mname, sname, "done", flush=True)
        del model
        torch.cuda.empty_cache()
    names = list(models)
    res: dict = {}
    mean_diffs: dict = {}
    for sname in sets:
        res[sname] = {}
        hb, db = store[(names[0], sname)]
        for other in names[1:]:
            ho, do = store[(other, sname)]
            L = db.shape[1]
            by_role: dict = {}
            for (xb, roles), (xo, _) in zip(hb, ho):
                d = (xo.float() - xb.float()).norm(dim=-1)  # [L, K]
                nb = xb.float().norm(dim=-1)
                cs = torch.nn.functional.cosine_similarity(xo.float(), xb.float(), dim=-1)  # [L, K]
                for r in set(roles):
                    mask = torch.tensor([x == r for x in roles])
                    num, den, csum, cnt = by_role.setdefault(r, [np.zeros(L), np.zeros(L), np.zeros(L), np.zeros(1)])
                    num += d[:, mask].sum(1).numpy()
                    den += nb[:, mask].sum(1).numpy()
                    csum += cs[:, mask].sum(1).numpy()
                    cnt += int(mask.sum())
            md = (do - db).mean(0)  # [L, d] mean change at the decision token
            mean_diffs[(sname, other)] = md
            res[sname][f"{other} - {names[0]}"] = {
                "rel_by_role": {r: (v[0] / v[1]).tolist() for r, v in by_role.items()},
                "cos_by_role": {r: (v[2] / v[3]).tolist() for r, v in by_role.items()},
                "tokens_by_role": {r: int(v[3][0]) for r, v in by_role.items()},
                "rel_decision": (
                    np.linalg.norm(do - db, axis=-1).mean(0) / np.linalg.norm(db, axis=-1).mean(0)
                ).tolist(),
                "mean_diff_norm": np.linalg.norm(md, axis=-1).tolist(),
                "mean_diff_share": (np.linalg.norm(md, axis=-1) / np.linalg.norm(do - db, axis=-1).mean(0)).tolist(),
            }
        if len(names) > 2:  # seed-to-seed: how far apart are two SFT runs on the same data?
            h1, d1 = store[(names[1], sname)]
            h2, d2 = store[(names[2], sname)]
            res[sname][f"{names[2]} - {names[1]}"] = {
                "rel_decision": (
                    np.linalg.norm(d2 - d1, axis=-1).mean(0) / np.linalg.norm(d1, axis=-1).mean(0)
                ).tolist()
            }

    def cos(a, b):
        return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    cosines: dict = {}
    snames = list(sets)
    for other in names[1:]:
        for i, s1 in enumerate(snames):
            for s2 in snames[i + 1 :]:
                a, b = mean_diffs[(s1, other)], mean_diffs[(s2, other)]
                cosines[f"{other}: {s1} vs {s2}"] = [cos(a[l], b[l]) for l in range(a.shape[0])]
    if len(names) > 2:
        for s in snames:
            a, b = mean_diffs[(s, names[1])], mean_diffs[(s, names[2])]
            cosines[f"seeds: {names[1]} vs {names[2]} on {s}"] = [cos(a[l], b[l]) for l in range(a.shape[0])]
    d = next(iter(mean_diffs.values())).shape[-1]
    r = np.random.default_rng(0).standard_normal((1000, 2, d))
    rc = np.abs((r[:, 0] * r[:, 1]).sum(-1) / (np.linalg.norm(r[:, 0], axis=-1) * np.linalg.norm(r[:, 1], axis=-1)))
    cosines["random |cos| 99th percentile"] = float(np.percentile(rc, 99))
    res["cosines"] = cosines
    res["n_prompts"] = {s: len(p) for s, p in sets.items()}
    return res


@app.function(
    image=image, gpu=["L40S", "A100-80GB"], secrets=[hf_secret], volumes={"/cache": cache}, timeout=3 * 60 * 60
)
def run_probe(models: dict[str, str], contexts: list[dict], tools: list, max_new: int = 160) -> dict:
    import numpy as np
    import torch

    out: dict = {"acts": {}, "gens": []}
    for mname, mpath in models.items():
        tok, model = _load(mpath)
        acts = []
        for k, cx in enumerate(contexts):
            row = {}
            for v, msgs in cx["variants"].items():
                ids = _ids(tok, msgs, tools)
                x = torch.tensor([ids], device="cuda")
                with torch.no_grad():
                    o = model(input_ids=x, output_hidden_states=True, logits_to_keep=1)
                    row[v] = torch.stack([h[0, -1] for h in o.hidden_states[1:]]).float().cpu().numpy()
                    if cx["split"] == "test":  # behaviour is scored on the test contexts only
                        g = model.generate(x, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.0)
                        out["gens"].append({"model": mname, "index": cx["index"], "split": cx["split"],
                                            "condition": v, "kind": cx["kind"], "cmd": cx["cmd"],
                                            "text": tok.decode(g[0, len(ids):], skip_special_tokens=False)})  # fmt: skip
            acts.append(row)
            if k % 10 == 0:
                print(mname, k, flush=True)
        out["acts"][mname] = {v: np.stack([a[v] for a in acts]).tolist() for v in ("success", "error", "decoy")}
        del model
        torch.cuda.empty_cache()
    return out


@app.function(
    image=image, gpu=["L40S", "A100-80GB"], secrets=[hf_secret], volumes={"/cache": cache}, timeout=3 * 60 * 60
)
def run_ablate(model_path: str, contexts: list[dict], tools: list, directions: dict[str, list[float]],
               max_new: int = 160) -> list[dict]:  # fmt: skip
    import torch

    tok, model = _load(model_path)
    layers = _layers(model)
    gens = []
    for dname, vec in directions.items():
        w = torch.tensor(vec, device="cuda", dtype=torch.float32)
        w = w / w.norm()

        def hook(_m, _i, o, w=w):
            h = o[0] if isinstance(o, tuple) else o
            h2 = (h.float() - (h.float() @ w)[..., None] * w).to(h.dtype)
            return (h2, *o[1:]) if isinstance(o, tuple) else h2

        handles = [layer.register_forward_hook(hook) for layer in layers]
        for cx in contexts:
            ids = _ids(tok, cx["variants"]["error"], tools)
            with torch.no_grad():
                g = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=max_new, do_sample=False,
                                   repetition_penalty=1.0)  # fmt: skip
            gens.append({"direction": dname, "index": cx["index"], "kind": cx["kind"], "cmd": cx["cmd"],
                         "text": tok.decode(g[0, len(ids):], skip_special_tokens=False)})  # fmt: skip
        for h in handles:
            h.remove()
        print(dname, "done", flush=True)
    return gens


@app.function(
    image=image, gpu=["L40S", "A100-80GB"], secrets=[hf_secret], volumes={"/cache": cache}, timeout=2 * 60 * 60
)
def run_steer(model_path: str, contexts: list[dict], tools: list, layer: int, directions: dict[str, list[float]],
              scales: list[float], max_new: int = 160) -> list[dict]:  # fmt: skip
    """Sufficiency test: add scale * unit direction to the residual stream after one layer, at every position, while
    the model writes its next turn after a successful call. Random directions get the same scale."""
    import torch

    tok, model = _load(model_path)
    target = _layers(model)[layer]
    gens = []
    for dname, vec in directions.items():
        w = torch.tensor(vec, device="cuda", dtype=torch.float32)
        w = w / w.norm()
        for scale in scales:

            def hook(_m, _i, o, w=w, scale=scale):
                h = o[0] if isinstance(o, tuple) else o
                h2 = (h.float() + scale * w).to(h.dtype)
                return (h2, *o[1:]) if isinstance(o, tuple) else h2

            handle = target.register_forward_hook(hook)
            for cx in contexts:
                ids = _ids(tok, cx["variants"]["success"], tools)
                with torch.no_grad():
                    g = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=max_new, do_sample=False,
                                       repetition_penalty=1.0)  # fmt: skip
                gens.append({"direction": dname, "scale": scale, "index": cx["index"], "kind": cx["kind"],
                             "cmd": cx["cmd"], "text": tok.decode(g[0, len(ids):], skip_special_tokens=False)})  # fmt: skip
            handle.remove()
            print(dname, scale, "done", flush=True)
    return gens


# ------------------------------------------------------------------------------------------------ local analysis
def classify(text: str, failing_cmd: str) -> str:
    """answer_now (writes answer.txt or ends without a tool call), repeat (same command) or new_call."""
    calls = re.findall(r"<tool_call>(.*?)(?:</tool_call>|$)", text, flags=re.S)
    if not calls:
        return "answer_now"
    first = calls[0]
    if ANSWER.search(first):
        return "answer_now"
    cmd = re.search(r"<parameter=command>\s*(.*?)\s*(?:</parameter>|$)", first, flags=re.S)
    if cmd and cmd.group(1).strip() == failing_cmd.strip():
        return "repeat"
    return "new_call"


def mentions_error(text: str) -> bool:
    pre = text.split("<tool_call>")[0].lower()
    return bool(re.search(r"error|fail|not found|missing|doesn't exist|does not exist|fix|instead|typo|wrong", pre))


@app.local_entrypoint()
def main(stage: str, sft: str = "albertlieadrian/at-qwen35-2b-sft500-s42", sft2: str = "", n: int = 60):
    tools = json.loads((HERE / "data" / "contexts" / "tools.json").read_text())
    outdir = HERE / "results" / "interp"
    outdir.mkdir(parents=True, exist_ok=True)
    if stage == "diff":
        models = {"base": BASE, "sft": sft} | ({"sft2": sft2} if sft2 else {})
        res = run_diff.remote(models, diff_prompts(n), tools)
        (outdir / "diff.json").write_text(json.dumps(res, indent=1))
        print("wrote results/interp/diff.json")
    elif stage == "probe":
        ctx = probe_contexts("eval", n) + probe_contexts("test", n)
        print(f"{len(ctx)} contexts")
        res = run_probe.remote({"base": BASE, "sft": sft}, ctx, tools)
        import numpy as np

        np.savez_compressed(outdir / "probe_acts.npz",
                            **{f"{m}_{v}": np.array(a, dtype=np.float32) for m, d in res["acts"].items()
                               for v, a in d.items()})  # fmt: skip
        meta = [{k: c[k] for k in ("index", "split", "cut", "kind", "cmd")} for c in ctx]
        (outdir / "probe_contexts.json").write_text(json.dumps(meta, indent=1))
        with open(outdir / "gens.jsonl", "w") as f:
            for g in res["gens"]:
                g["action"] = classify(g["text"], g["cmd"])
                g["mentions_error"] = mentions_error(g["text"])
                f.write(json.dumps(g) + "\n")
        print("saved probe activations and", len(res["gens"]), "generations")
    elif stage == "ablate":
        import numpy as np

        d = json.loads((outdir / "probe.json").read_text())
        dirs = {"probe": d["direction"]}
        rng = np.random.default_rng(1)
        for k in range(4):
            dirs[f"random{k}"] = rng.standard_normal(len(d["direction"])).tolist()
        gens = run_ablate.remote(sft, probe_contexts("test", n), tools, dirs)
        with open(outdir / "ablate_gens.jsonl", "w") as f:
            for g in gens:
                g["action"] = classify(g["text"], g["cmd"])
                g["mentions_error"] = mentions_error(g["text"])
                f.write(json.dumps(g) + "\n")
        print("saved", len(gens), "ablation generations")
    elif stage == "steer":
        import numpy as np

        d = json.loads((outdir / "probe.json").read_text())
        dirs = {"probe": d["direction"]}
        rng = np.random.default_rng(2)
        for k in range(3):
            dirs[f"random{k}"] = rng.standard_normal(len(d["direction"])).tolist()
        gap = d["direction_norm"]["sft"]  # error-minus-success distance along the direction at that layer
        scales = [gap * m for m in (1.0, 4.0, 16.0)]
        gens = run_steer.remote(sft, probe_contexts("test", n), tools, d["layer"], dirs, scales)
        with open(outdir / "steer_gens.jsonl", "w") as f:
            for g in gens:
                g["action"] = classify(g["text"], g["cmd"])
                g["mentions_error"] = mentions_error(g["text"])
                f.write(json.dumps(g) + "\n")
        print("saved", len(gens), "steering generations; scales", scales)
