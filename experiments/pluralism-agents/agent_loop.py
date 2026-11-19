"""Multi-turn tau-bench rollouts, judge rewards, GRPO updates and the drift readout, for one Modal container.

One container holds everything (an H200):
    vLLM server (shared)    : Qwen3-30B-A3B-Instruct-2507, the user simulator and every training judge
    vLLM engine in-process  : Qwen3-8B policy with the current LoRA adapter
    PyTorch model           : Qwen3-8B + LoRA, the GRPO update and the KL drift readout
    tau_server.py subprocess: the tau-bench tools, database and ground-truth reward (its own venv)

Token bookkeeping is exact: each conversation is one growing list of token ids. The agent's generated tokens are
appended as they were sampled (mask 1); user messages and tool results are appended as rendered by Qwen3's chat
template (mask 0). Sampling and training see the same tokens, so the GRPO update uses exactly what was sampled.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GREETING = "Hi! How can I help you today?"
STOP_TOKENS = ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###")
IM_END = 151645
ENDOFTEXT = 151643
GEN_PROMPT = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
SIM_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
POLICY_MODEL = "Qwen/Qwen3-8B"


# ----------------------------------------------------------------------------- services


class TauEnv:
    """The tau-bench environment in its own Python process (see tau_server.py)."""

    def __init__(self, python="/opt/tau/bin/python", script="/root/tau_server.py"):
        self.p = subprocess.Popen([python, script], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def batch(self, calls):
        if not calls:
            return []
        self.p.stdin.write(json.dumps({"calls": calls}) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("tau_server died")
        return json.loads(line)["results"]


def start_sim_server(gpu_util=0.30, port=8001, max_len=32768, device=None):
    cmd = [
        "vllm", "serve", SIM_MODEL, "--port", str(port), "--served-model-name", "sim",
        "--gpu-memory-utilization", str(gpu_util), "--max-model-len", str(max_len),
        "--enable-prefix-caching", "--max-num-seqs", "128", "--disable-log-requests",
    ]  # fmt: skip
    log = open("/tmp/sim_server.log", "w")
    env = dict(os.environ)
    if device is not None:  # two-GPU containers: the simulator gets its own GPU
        env["CUDA_VISIBLE_DEVICES"] = str(device)
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    import requests

    t0 = time.time()
    while time.time() - t0 < 1500:
        if p.poll() is not None:
            full = open("/tmp/sim_server.log").read()
            errs = [ln for ln in full.splitlines() if "Error" in ln or "error" in ln or "CUDA" in ln]
            print("\n".join(errs[-40:]), flush=True)
            raise RuntimeError("sim server exited:\n" + full[-3000:])
        try:
            if requests.get(f"http://localhost:{port}/health", timeout=2).status_code == 200:
                print(f"sim server up in {time.time() - t0:.0f}s", flush=True)
                return p
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError("sim server did not start")


class Chat:
    """Concurrent chat completions against the simulator/judge server (user simulator and judges)."""

    def __init__(self, url="http://localhost:8001", key=None, workers=96):
        import requests

        self.s = requests.Session()
        for pre in ("http://", "https://"):
            self.s.mount(pre, requests.adapters.HTTPAdapter(pool_maxsize=workers))
        self.url = url.rstrip("/") + "/v1/chat/completions"
        self.headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.pool = ThreadPoolExecutor(workers)
        self.tokens = 0

    def one(self, messages, temperature=0.0, max_tokens=400):
        for attempt in range(6):
            try:
                r = self.s.post(
                    self.url,
                    headers=self.headers,
                    json={"model": "sim", "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
                    timeout=600,
                )
                r.raise_for_status()
                j = r.json()
                self.tokens += j["usage"]["total_tokens"]
                return j["choices"][0]["message"]["content"] or ""
            except Exception as e:
                if attempt == 5:
                    print("chat failed:", repr(e)[:300], flush=True)
                    return ""
                time.sleep(3 * (attempt + 1))

    def many(self, list_of_messages, **kw):
        return list(self.pool.map(lambda m: self.one(m, **kw), list_of_messages))


# ----------------------------------------------------------------------------- conversations


def parse_agent(text):
    """Returns (tool_calls, message). tool_calls: list of (name, arguments or None if unparseable, raw)."""
    blocks = TOOL_RE.findall(text)
    if not blocks:
        return [], text.strip()
    calls = []
    for b in blocks:
        try:
            j = json.loads(b)
            args = j.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            calls.append((str(j["name"]), args if isinstance(args, dict) else None, b))
        except Exception:
            calls.append((None, None, b))
    return calls, ""


class Conv:
    def __init__(self, key, domain, task, sample):
        self.key, self.domain, self.task, self.sample = key, domain, task, sample
        self.ids, self.mask = [], []
        self.sim_msgs = [{"role": "system", "content": task["user_system"]}, {"role": "user", "content": GREETING}]
        self.transcript = []
        self.turn = "user"
        self.done = False
        self.termination = None
        self.n_agent_turns = 0
        self.n_errors = 0
        self.plan = []  # tool names in call order
        self.writes = []  # [name, args, ok]
        self.agent_texts = []
        self.calls = []  # "name(args)" strings, for the training-time loop stop

    def append(self, tok, text, gen):
        ids = tok.encode(text, add_special_tokens=False) if isinstance(text, str) else text
        self.ids += ids
        self.mask += [gen] * len(ids)

    def finish(self, why):
        self.done, self.termination = True, why

    def summary(self):
        return {
            "key": self.key,
            "domain": self.domain,
            "task_id": self.task["id"],
            "sample": self.sample,
            "termination": self.termination,
            "n_agent_turns": self.n_agent_turns,
            "n_errors": self.n_errors,
            "n_tokens": len(self.ids),
            "n_agent_tokens": sum(self.mask),
            "plan": self.plan,
            "writes": self.writes,
            "transcript": self.transcript,
        }


def rollout(llm, tok, tau, chat, D, items, lora=None, temperature=1.0, top_p=1.0, top_k=-1, seed=0,
            max_turns=40, max_ctx=24000, max_new=1024, tag="r", loop_limit=None):  # fmt: skip
    """items: list of (domain, task, sample_index). Returns list of Conv, scored with the ground-truth reward."""
    from vllm import SamplingParams, TokensPrompt

    convs = [Conv(f"{tag}-{i}", d, t, s) for i, (d, t, s) in enumerate(items)]
    tau.batch([{"op": "new", "key": c.key, "domain": c.domain} for c in convs])
    write_tools = {d: set(D[d]["write_tools"]) for d in D}
    t0 = time.time()
    rounds = 0
    while True:
        # user phase
        users = [c for c in convs if not c.done and c.turn == "user"]
        if users:
            replies = chat.many([c.sim_msgs for c in users], temperature=0.0, max_tokens=400)
            for c, r in zip(users, replies):
                r = r.strip()
                c.sim_msgs.append({"role": "assistant", "content": r})
                visible = r
                for s in STOP_TOKENS:
                    visible = visible.replace(s, "").strip()
                if visible:
                    c.transcript.append({"role": "user", "content": visible})
                if any(s in r for s in STOP_TOKENS):
                    c.finish("user_stop")
                    continue
                if not c.ids:  # first turn: system + tools + greeting + first user message
                    msgs = [
                        {"role": "system", "content": D[c.domain]["agent_system"]},
                        {"role": "assistant", "content": GREETING},
                        {"role": "user", "content": r},
                    ]
                    s = tok.apply_chat_template(
                        msgs, tools=D[c.domain]["tools"], tokenize=False, add_generation_prompt=True,
                        enable_thinking=False,
                    )  # fmt: skip
                    c.append(tok, s, 0)
                else:
                    c.append(tok, f"\n<|im_start|>user\n{r}<|im_end|>\n{GEN_PROMPT}", 0)
                c.turn = "agent"
        agents = [c for c in convs if not c.done and c.turn == "agent"]
        if not [c for c in convs if not c.done]:
            break
        if agents:
            params = [
                SamplingParams(
                    n=1,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_tokens=max_new,
                    repetition_penalty=1.0,
                    seed=(seed * 1_000_003 + rounds * 1009 + i) % (2**31),
                    stop_token_ids=[IM_END, ENDOFTEXT],
                )
                for i in range(len(agents))
            ]
            outs = llm.generate(
                [TokensPrompt(prompt_token_ids=c.ids) for c in agents], params, lora_request=lora, use_tqdm=False
            )
            tool_jobs = []  # (conv, [(name, args, raw)])
            for c, o in zip(agents, outs):
                g = o.outputs[0]
                ids = list(g.token_ids)
                while ids and ids[-1] in (IM_END, ENDOFTEXT):
                    ids.pop()
                c.n_agent_turns += 1
                c.append(tok, [*ids, IM_END], 1)
                if g.finish_reason == "length":
                    c.finish("agent_len")
                    continue
                text = tok.decode(ids, skip_special_tokens=False)
                calls, msg = parse_agent(text)
                if calls:
                    tool_jobs.append((c, calls))
                else:
                    c.transcript.append({"role": "agent", "content": msg})
                    c.agent_texts.append(msg)
                    c.sim_msgs.append({"role": "user", "content": msg})
                    c.turn = "user"
                if c.n_agent_turns >= max_turns and not c.done:
                    c.finish("max_turns")
            # execute tool calls in one batch (even for a conversation that just hit max_turns, so the DB is right)
            req = []
            for c, calls in tool_jobs:
                for name, args, _raw in calls:
                    if name is not None and args is not None:
                        req.append({"op": "call", "key": c.key, "name": name, "arguments": args})
            res = iter(tau.batch(req))
            for c, calls in tool_jobs:
                outs_txt = []
                for name, args, raw in calls:
                    if name is None or args is None:
                        content, err = f"Error: could not parse tool call: {raw[:200]}", True
                        c.plan.append("<invalid>")
                    else:
                        r = next(res)
                        content = r.get("content", r.get("error_msg", "Error"))
                        err = bool(r.get("error", "error_msg" in r))
                        c.plan.append(name)
                        if name in write_tools[c.domain]:
                            c.writes.append([name, args, not err])
                    c.n_errors += err
                    shown = json.dumps(args) if args is not None else raw
                    c.calls.append(f"{name}({shown})")
                    c.transcript.append({"role": "tool_call", "content": f"{name}({shown})"})
                    c.transcript.append({"role": "tool_result", "content": content})
                    outs_txt.append(f"<tool_response>\n{content}\n</tool_response>")
                if c.done:
                    continue
                c.append(tok, "\n<|im_start|>user\n" + "\n".join(outs_txt) + f"<|im_end|>\n{GEN_PROMPT}", 0)
                if c.n_errors >= 10:
                    c.finish("too_many_errors")
                elif loop_limit and len(c.calls) >= loop_limit and len(set(c.calls[-loop_limit:])) == 1:
                    c.finish("loop")  # training only: the same call repeated loop_limit times in a row
        for c in convs:
            if not c.done and len(c.ids) > max_ctx:
                c.finish("max_ctx")
        rounds += 1
        if rounds > 4 * max_turns:
            for c in convs:
                if not c.done:
                    c.finish("max_rounds")
    # ground-truth reward (tau-bench gives 0 to any conversation that did not end normally)
    res = tau.batch(
        [{"op": "score", "key": c.key, "task": c.task, "agent_texts": c.agent_texts} for c in convs]
        + [{"op": "close", "key": c.key} for c in convs]
    )
    for c, r in zip(convs, res):
        ok = c.termination == "user_stop"
        c.reward = float(r["reward"]) if ok else 0.0
        c.db = float(r["db"]) if ok else 0.0
        c.communicate = float(r["communicate"])
    print(
        f"rollout {tag}: {len(convs)} convs, {rounds} rounds, {time.time() - t0:.0f}s, "
        f"success {sum(c.reward for c in convs) / len(convs):.3f}, "
        f"mean tokens {sum(len(c.ids) for c in convs) / len(convs):.0f}",
        flush=True,
    )
    return convs


def policy_text(D, domain):
    s = D[domain]["agent_system"]
    return s[s.index("<policy>") + len("<policy>") : s.index("</policy>")].strip()


def judge_scores(chat, D, convs, arms=("single", "jury")):
    """Scores every conversation with each arm's judges. Conversations that did not end normally get the floor (1)
    without a judge call. Returns per-conv dict {arm: {"reward": float, "scores": {...}, "raw": {...}}}."""
    sys.path.insert(0, "/root")
    import judges

    jobs, index = [], []
    out = [{} for _ in convs]
    for i, c in enumerate(convs):
        conv = c if isinstance(c, dict) else c.summary()
        for arm in arms:
            if conv["termination"] != "user_stop":
                out[i][arm] = {"reward": 1.0, "scores": {}, "raw": {}, "floor": True}
                continue
            for name, p in judges.prompts_for(arm, conv, policy_text(D, conv["domain"])).items():
                jobs.append([{"role": "user", "content": p}])
                index.append((i, arm, name))
    texts = chat.many(jobs, temperature=0.0, max_tokens=300)
    for (i, arm, name), t in zip(index, texts):
        d = out[i].setdefault(arm, {"scores": {}, "raw": {}})
        d["scores"][name] = judges.parse_score(t)
        d["raw"][name] = t
    for i in range(len(convs)):
        for arm in arms:
            d = out[i][arm]
            if "reward" not in d:
                d["reward"] = judges.aggregate(arm, d["scores"])
    return out


# ----------------------------------------------------------------------------- training pieces


def build_trainer(model_name, rank, lr, seed):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    lcfg = LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )  # fmt: skip
    model = get_peft_model(model, lcfg)
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    return model, params, opt


def agent_logprobs(model, ids, mask):
    """Per-token log-probs of the agent's tokens only (logits computed only where needed)."""
    import torch

    pos = [t - 1 for t in range(1, len(ids)) if mask[t]]
    x = torch.tensor([ids], device="cuda")
    p = torch.tensor(pos, device="cuda")
    logits = model(input_ids=x, logits_to_keep=p).logits[0].float()
    tgt = x[0, p + 1]
    return logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)


def grpo_step(model, params, opt, seqs, advs, n_tok):
    """seqs: list of (ids, mask); advs: per sequence. Loss = -sum(adv * logp) / all agent tokens in the batch."""
    import torch

    model.train()
    opt.zero_grad(set_to_none=True)
    loss_val = 0.0
    for (ids, mask), a in zip(seqs, advs):
        lp = agent_logprobs(model, ids, mask)
        loss = -(a * lp).sum() / max(n_tok, 1)
        loss.backward()
        loss_val += float(loss.detach())
    gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
    if gnorm != gnorm or gnorm == float("inf"):
        opt.zero_grad(set_to_none=True)
        return loss_val, gnorm, False
    opt.step()
    return loss_val, gnorm, True


def drift_kl(model, neutral):
    """Mean per-token KL(base || trained), teacher-forced on fixed base-model samples. neutral: [(prompt_ids, reply_ids)]."""
    import torch

    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for p, r in neutral:
            if not r:
                continue
            x = torch.tensor([p + r], device="cuda")
            pos = torch.arange(len(p) - 1, len(p) + len(r) - 1, device="cuda")
            with model.disable_adapter():
                lb = model(input_ids=x, logits_to_keep=pos).logits[0].float().log_softmax(-1)
            lt = model(input_ids=x, logits_to_keep=pos).logits[0].float().log_softmax(-1)
            tot += float((lb.exp() * (lb - lt)).sum(-1).sum())
            n += len(r)
    return tot / max(n, 1)


def group_advantages(rewards, eps=1e-6):
    m = sum(rewards) / len(rewards)
    sd = (sum((r - m) ** 2 for r in rewards) / len(rewards)) ** 0.5
    if sd < eps:
        return [0.0] * len(rewards)
    return [(r - m) / sd for r in rewards]


def conv_record(c, extra=None):
    d = c.summary()
    d.update({"reward": c.reward, "db": c.db, "communicate": c.communicate})
    if extra:
        d.update(extra)
    return d


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r) for r in rows))


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
