"""FIXED. The improver's only way out of its sandbox: JSON requests to the harness over two inherited pipes.

    from broker import ask, validate
    text = ask(messages, max_tokens=8000, temperature=0.7)   # one LLM call; the harness meters tokens and caps them
    res = validate({"recipe.py": src})                       # the harness builds the recipe in its own sandbox

The harness, not this file, decides the model, enforces the token caps and runs the validation, so editing or bypassing
this client gains nothing: a raw request on the pipe goes through the same checks.
"""

from __future__ import annotations

import json
import os

_REQ = os.fdopen(int(os.environ["RSI_REQ_FD"]), "w", buffering=1)
_REP = os.fdopen(int(os.environ["RSI_REP_FD"]), "r")


def _call(req: dict) -> dict:
    _REQ.write(json.dumps(req) + "\n")
    _REQ.flush()
    line = _REP.readline()
    if not line:
        raise RuntimeError("harness closed the pipe")
    return json.loads(line)


def ask(messages: list[dict], max_tokens: int = 8000, temperature: float = 0.7) -> str:
    """One chat completion. Raises RuntimeError when the harness refuses (cap reached) or the provider fails."""
    r = _call({"op": "llm", "messages": messages, "max_tokens": max_tokens, "temperature": temperature})
    if r.get("error"):
        raise RuntimeError(r["error"])
    return r["content"]


def validate(files: dict[str, str]) -> dict:
    """Check proposed files without training: a recipe is built in a sandbox and its rows checked; improver files are
    compiled. Returns {"ok": bool, "error": str | None, "summary": {...}}. At most 3 calls per proposal."""
    return _call({"op": "validate", "files": files})
