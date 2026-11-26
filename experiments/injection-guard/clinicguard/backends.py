"""Model backends. Each takes a batch of conversations and returns one output string per conversation.

- stub: a scripted always-comply model for CPU tests. It plays each case's `stub` plan, the steps a model that
  obeys every instruction it can see would take. A step marked `when_visible` runs only if its marker text is in
  what the model can see, so an instruction hidden by a guard is not followed.
- vllm: an open model in-process (used inside the Modal container, see modal_app.py).
- openai: any OpenAI-compatible endpoint (for example `vllm serve` on Modal), standard library only.
- anthropic: a placeholder that refuses to run; plug a client in here once running on that provider is allowed.
"""

from __future__ import annotations

import base64
import codecs
import json
import re
import urllib.request

DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def _filters(value: str, f: str) -> str:
    if f == "b64":
        return base64.b64encode(value.encode()).decode()
    if f == "hex":
        return value.encode().hex()
    if f == "rot13":
        return codecs.encode(value, "rot13")
    if f == "spaced":
        return " ".join(value)
    if f == "reversed":
        return value[::-1]
    if f == "leet":
        return value.translate(str.maketrans("aeiost", "43105+"))
    if f == "digitwords":
        return " ".join(DIGITS[int(ch)] if ch.isdigit() else ("dash" if ch == "-" else ch) for ch in value)
    return value


def _yesno(last: str) -> str:
    try:
        obj = json.loads(last)
    except json.JSONDecodeError:
        return "I can't check that."
    if "error" in obj:
        return "I can't check that."
    rows = obj.get("rows") or [[0]]
    return "Yes." if rows and rows[0] and rows[0][0] else "No."


def fill(template, conv):
    if isinstance(template, dict):
        return {k: fill(v, conv) for k, v in template.items()}
    if not isinstance(template, str):
        return template

    def sub(m):
        var, _, f = m.group(1).partition("|")
        if var == "last":
            val = conv.last_tool_result()
        elif var == "syskey":
            val = conv.env.secrets["system_key"]
        elif var == "yesno":
            return _yesno(conv.last_tool_result())
        else:
            val = m.group(0)
        return _filters(val, f)

    return re.sub(r"\{\{(\w+(?:\|\w+)?)\}\}", sub, template)


class StubBackend:
    name = "stub"

    def generate(self, convs):
        return [self._one(c) for c in convs]

    def _one(self, conv):
        plan = conv.case["turns"][conv.turn].get("stub", []) if conv.turn < len(conv.case["turns"]) else []
        visible = conv.visible_text()
        while conv.stub_pos < len(plan):
            step = plan[conv.stub_pos]
            conv.stub_pos += 1
            if "when_visible" in step and step["when_visible"] not in visible:
                continue
            if "tool" in step:
                return "TOOL: " + json.dumps({"name": step["tool"], "args": fill(step.get("args", {}), conv)})
            return fill(step["reply"], conv)
        return "Done."


class VLLMBackend:
    """In-process vLLM. Build once per model; generate() batches every pending conversation."""

    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 384, max_model_len: int = 8192):
        from vllm import LLM, SamplingParams

        self.name = model
        self.llm = LLM(
            model,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.9,
            enable_prefix_caching=True,
            generation_config="vllm",
            seed=0,
        )
        self.tok = self.llm.get_tokenizer()
        # Unseeded per request, so samples of the same case differ; repetition_penalty 1.0 is explicit.
        self.params = SamplingParams(
            temperature=temperature, top_p=0.95, max_tokens=max_tokens, repetition_penalty=1.0, stop=["\nTool result"]
        )

    def generate(self, convs):
        texts = [
            self.tok.apply_chat_template(c.messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for c in convs
        ]
        outs = self.llm.generate(texts, self.params, use_tqdm=False)
        return [re.sub(r"<think>.*?</think>", "", o.outputs[0].text, flags=re.DOTALL).strip() for o in outs]


class OpenAICompatBackend:
    def __init__(self, base_url: str, model: str, temperature: float = 0.7, api_key: str = "none"):
        self.name, self.url, self.model, self.t, self.key = model, base_url.rstrip("/"), model, temperature, api_key

    def generate(self, convs):
        outs = []
        for c in convs:
            body = json.dumps(
                {"model": self.model, "messages": c.messages, "temperature": self.t, "max_tokens": 384}
            ).encode()
            req = urllib.request.Request(
                self.url + "/chat/completions",
                data=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"},
            )
            with urllib.request.urlopen(req, timeout=300) as r:
                outs.append(json.loads(r.read())["choices"][0]["message"]["content"] or "")
        return outs


class AnthropicBackend:
    name = "anthropic"

    def generate(self, convs):
        raise NotImplementedError("Not wired up: add a client here once running against this provider is authorised.")


def make_backend(name: str, **kw):
    if name == "stub":
        return StubBackend()
    if name == "vllm":
        return VLLMBackend(**kw)
    if name == "openai":
        return OpenAICompatBackend(**kw)
    if name == "anthropic":
        return AnthropicBackend()
    raise ValueError(name)
