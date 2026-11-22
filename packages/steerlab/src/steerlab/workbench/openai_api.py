"""The OpenAI-compatible chat API: request and response shapes, so any OpenAI client can talk to a steered model.

    POST /v1/chat/completions     {"model": "...", "messages": [...], "max_tokens": 64, "temperature": 0,
                                   "steer": "4 french.english", "edits": ["my-edit"], "probes": ["formal.casual"]}
    GET  /v1/models               the one loaded model

Point the official client at the inspect server and pass the extras through `extra_body`:

    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:8770/v1", api_key="unused")
    client.chat.completions.create(model="Qwen/Qwen2.5-0.5B-Instruct", messages=[{"role": "user", "content": "Hi"}],
                                   extra_body={"steer": [{"probe": "formal.casual", "strength": 3}]})

The extras: `steer` (an expression or a list of entries, see `steerlab.workbench.session`), `edits` (edit artifact
names or inline edit dicts) and `probes` (names, or `true` for every probe). With `probes` set, the response carries
`steerlab.probes`: each probe's per-token z-scores, their mean, and the random-direction band beside it. Supported
standard fields: `max_tokens` / `max_completion_tokens`, `temperature` (default 1, as OpenAI), `top_p`, `seed`,
`stop`, `stream`, `logprobs` + `top_logprobs` (at most 5), `n` = 1. Unknown fields are ignored, as OpenAI-compatible
servers usually do.

Pydantic only; no torch.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from steerlab.workbench.session import MAX_NEW_TOKENS, Sampling


class ContentPart(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str = "text"
    text: str = ""


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[ContentPart] | None = ""

    def text(self) -> str:
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        return "".join(part.text for part in self.content if part.type == "text")


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1, max_length=200)
    max_tokens: int | None = Field(default=None, ge=1, le=MAX_NEW_TOKENS)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=MAX_NEW_TOKENS)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    n: int = Field(default=1, ge=1, le=1)
    logprobs: bool = False
    top_logprobs: int = Field(default=0, ge=0, le=5)
    # steerlab extras
    steer: str | list[dict[str, Any]] | None = None
    edits: list[str | dict[str, Any]] | None = None
    probes: bool | list[str] | None = None

    @field_validator("stop")
    @classmethod
    def _stops(cls, value: str | list[str] | None) -> str | list[str] | None:
        if isinstance(value, list) and len(value) > 4:
            raise ValueError("at most 4 stop sequences")
        return value

    def chat_messages(self) -> list[dict[str, str]]:
        """Messages for the chat template (`developer` is OpenAI's newer name for `system`)."""
        return [{"role": "system" if m.role == "developer" else m.role, "content": m.text()} for m in self.messages]

    def sampling(self) -> Sampling:
        stops = [self.stop] if isinstance(self.stop, str) else list(self.stop or [])
        return Sampling(
            max_tokens=self.max_completion_tokens or self.max_tokens or 256,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            stop=tuple(stops),
        )

    def probe_names(self, available: list[str]) -> list[str]:
        if self.probes is True:
            return list(available)
        if not self.probes:
            return []
        return [name for name in self.probes if name in available]


def completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _logprobs(tokens: list[dict[str, Any]], top: int) -> dict[str, Any]:
    return {
        "content": [
            {
                "token": t["token"],
                "logprob": t["logprob"],
                "bytes": list(t["token"].encode()),
                "top_logprobs": [
                    {"token": a["text"], "logprob": a["logprob"], "bytes": list(a["text"].encode())}
                    for a in t["alternatives"][:top]
                ],
            }
            for t in tokens
        ]
    }


def probe_summary(tokens: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    out = {}
    for name in names:
        z = [t["probes"][name] for t in tokens if name in t.get("probes", {})]
        band = [t["bands"][name] for t in tokens if name in t.get("bands", {})]
        out[name] = {
            "z": z,
            "mean": round(sum(z) / len(z), 4) if z else None,
            "random_band": round(sum(band) / len(band), 4) if band else None,
        }
    return out


def extras(result: dict[str, Any], probe_names: list[str]) -> dict[str, Any]:
    return {
        "steer": result.get("steer", []),
        "edits": result.get("edits", []),
        "messages": result.get("messages", []),
        "probes": probe_summary(result.get("tokens", []), probe_names) if probe_names else {},
    }


def finish_reason(result: dict[str, Any]) -> str:
    return "length" if result.get("finish_reason") == "length" else "stop"


def response(request: ChatRequest, result: dict[str, Any], model_id: str, probe_names: list[str]) -> dict[str, Any]:
    """The `chat.completion` object for one finished reply."""
    choice: dict[str, Any] = {
        "index": 0,
        "message": {"role": "assistant", "content": result["text"]},
        "finish_reason": finish_reason(result),
        "logprobs": _logprobs(result["tokens"], request.top_logprobs) if request.logprobs else None,
    }
    completion = len(result["token_ids"])
    return {
        "id": completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [choice],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": completion,
            "total_tokens": result["prompt_tokens"] + completion,
        },
        "steerlab": extras(result, probe_names),
    }


def chunk(cid: str, model_id: str, delta: dict[str, Any], finish: str | None = None, **more: Any) -> dict[str, Any]:
    """One `chat.completion.chunk` of a streamed reply."""
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": None}],
        **more,
    }


def models(model_id: str) -> dict[str, Any]:
    return {"object": "list", "data": [{"id": model_id, "object": "model", "created": 0, "owned_by": "steerlab"}]}
