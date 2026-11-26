"""The improver's LLM, metered (FIXED, runs outside the sandbox).

One model for both arms (rsi.IMPROVER_MODEL). Pilot 2: Qwen3-Coder-30B-A3B-Instruct (FP8) served by vLLM on Modal
(modal_llm.py), because OpenRouter refused payment (402); pilot 1 used Claude Sonnet 5.5 through OpenRouter key 2.
Sampling: the caller's temperature, top_p 0.8, top_k 20, repetition penalty 1.05 (Qwen3-Coder's recommended settings).
Caps, all checked before each call:
- per arm: rsi.ARM_TOKENS input and output tokens (equal for L2 and L3; L3's self-edits count against it);
- per proposal or self-edit: rsi.PROPOSAL_TOKENS;
- project: rsi.CAP_API dollars on the ledger.
Every call is one line in results/llm_calls.jsonl; the dollar total also goes to the spend ledger as kind "api".
"""

from __future__ import annotations

import threading
import time

from rsi import (
    ARM_TOKENS,
    CAP_API,
    IMPROVER_MODEL,
    IMPROVER_PRICE,
    LLM_APP,
    LLM_CONTEXT,
    PROPOSAL_TOKENS,
    RESULTS,
    append_jsonl,
    log_spend,
    read_jsonl,
    spent,
)

CALLS = RESULTS / "llm_calls.jsonl"
_LOCK = threading.Lock()
_CLIENT = None


_URL: list[str] = []
_READY = threading.Lock()


def _key() -> str:
    from modal_llm import llm_key

    return llm_key()


def _url() -> str:
    if not _URL:
        import modal

        _URL.append(modal.Cls.from_name(LLM_APP, "Server")().serve.get_web_url())
    return _URL[0]


def ready(wait: float = 1800) -> None:
    """Block until the server answers /v1/models (it scales to zero; a cold start takes minutes)."""
    import requests

    with _READY:
        t0 = time.time()
        while True:
            try:
                r = requests.get(_url() + "/v1/models", headers={"Authorization": f"Bearer {_key()}"}, timeout=90)
                if r.status_code == 200:
                    return
            except Exception:  # noqa: BLE001, S110 - not up yet
                pass
            if time.time() - t0 > wait:
                raise RuntimeError("improver LLM server did not come up")
            time.sleep(10)


def _client():
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI

        _CLIENT = OpenAI(base_url=_url() + "/v1", api_key=_key(), max_retries=0)
    return _CLIENT


def used(arm: str) -> dict:
    rows = [r for r in read_jsonl(CALLS) if r["arm"] == arm]
    return {"in": sum(r["in"] for r in rows), "out": sum(r["out"] for r in rows), "usd": sum(r["usd"] for r in rows)}


class Budget:
    """Token use of one proposal (or one self-edit)."""

    def __init__(self) -> None:
        self.inp = self.out = self.calls = 0


def chat(
    arm: str, node: str, purpose: str, messages: list[dict], max_tokens: int, temperature: float, budget: Budget
) -> dict:
    max_tokens = int(min(max(256, max_tokens), 16000))
    temperature = float(min(max(0.0, temperature), 1.0))
    if not isinstance(messages, list) or not all(
        isinstance(m, dict) and m.get("role") in ("system", "user", "assistant") and isinstance(m.get("content"), str)
        for m in messages
    ):
        return {"error": "messages must be a list of {role: system|user|assistant, content: str}"}
    est_in = sum(len(m["content"]) for m in messages) // 3  # a rough pre-check; the real count comes back from the API
    with _LOCK:
        u = used(arm)
        if u["in"] + est_in > ARM_TOKENS["in"] or u["out"] + max_tokens > ARM_TOKENS["out"]:
            return {"error": f"arm token cap reached ({u['in']} in, {u['out']} out of {ARM_TOKENS})"}
        if budget.inp + est_in > PROPOSAL_TOKENS["in"] or budget.out + max_tokens > PROPOSAL_TOKENS["out"]:
            return {"error": f"proposal token cap reached ({budget.inp} in, {budget.out} out of {PROPOSAL_TOKENS})"}
        if CAP_API > 0 and spent("api") >= CAP_API:  # pilot 2: no paid API (cap 0 means unused)
            return {"error": "project API cap reached"}
    t0 = time.time()
    err, text, finish, pin, pout = None, "", None, 0, 0
    max_tokens = max(256, min(max_tokens, LLM_CONTEXT - est_in - 1024))  # prompt + output must fit the context
    for attempt in range(4):
        try:
            ready()
            r = _client().chat.completions.create(
                model="improver",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.8,
                extra_body={"top_k": 20, "repetition_penalty": 1.05},
                timeout=900,
            )
            if r.usage:
                pin, pout = r.usage.prompt_tokens, r.usage.completion_tokens
            text = r.choices[0].message.content or ""
            finish = r.choices[0].finish_reason
            err = None
            break
        except Exception as e:  # noqa: BLE001 - provider errors are retried, then reported to the improver
            err = f"{type(e).__name__}: {str(e)[:300]}"
            if "402" in err:  # payment refused: never retry (lesson G8)
                break
            if "context length" in err or "maximum context" in err:  # the prompt is longer than estimated
                max_tokens = max(256, max_tokens // 2)
            time.sleep(3 * 2**attempt)
    usd = (pin * IMPROVER_PRICE[0] + pout * IMPROVER_PRICE[1]) / 1e6
    with _LOCK:
        budget.inp += pin
        budget.out += pout
        budget.calls += 1
        append_jsonl(
            CALLS,
            {
                "t": time.strftime("%Y-%m-%d %H:%M:%S"),
                "arm": arm,
                "node": node,
                "purpose": purpose,
                "in": pin,
                "out": pout,
                "usd": round(usd, 5),
                "seconds": round(time.time() - t0, 1),
                "finish": finish,
                "error": err,
                "model": IMPROVER_MODEL,
            },
        )
        if usd:
            log_spend(f"llm_{arm}_{node}", usd, kind="api", note="measured (token usage)")
    if err:
        return {"error": err}
    return {"content": text, "finish": finish, "usage": {"in": pin, "out": pout}}
