"""Guards and the levels that combine them.

A guard is a set of hooks the agent calls at fixed points. Every hook is deterministic code: no model runs inside a
guard. A hook that returns a string denies the step, and the string names the rule that fired.

    check_input(text, ctx)              before the model sees a user message (L1)
    on_tool_result(name, result, ctx)   rewrite a tool result before the model sees it (P1 quarantine)
    check_tool_call(name, args, ctx)    before a tool runs (provenance, budgets)
    transform_args(name, args, ctx)     after the check, before the tool runs (P1 fills its handles)
    transform_reply(text, ctx)          before egress checks on a reply (P1 fills its handles)
    check_egress(text, channel, ctx)    before a reply or an email leaves (L2, L3, P2 DLP)
    setup(env, ctx)                     once per conversation (the SQL firewall installs its authorizer here)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Ctx:
    env: object
    user_messages: list = field(default_factory=list)
    untrusted_texts: list = field(default_factory=list)
    handles: dict = field(default_factory=dict)
    counters: dict = field(default_factory=dict)
    events: list = field(default_factory=list)  # {"guard", "rule", "stage"} for every denial
    prompt_notes: list = field(default_factory=list)  # extra system-prompt lines a guard needs the model to know

    @property
    def current_user(self) -> str:
        return self.user_messages[-1] if self.user_messages else ""


class Guard:
    name = "guard"

    def setup(self, env, ctx: Ctx) -> None:
        pass

    def check_input(self, text: str, ctx: Ctx) -> str | None:
        return None

    def on_tool_result(self, name: str, result: dict, ctx: Ctx) -> dict:
        return result

    def check_tool_call(self, name: str, args: dict, ctx: Ctx) -> str | None:
        return None

    def transform_args(self, name: str, args: dict, ctx: Ctx) -> dict:
        return args

    def transform_reply(self, text: str, ctx: Ctx) -> str:
        return text

    def check_egress(self, text: str, channel: str, ctx: Ctx) -> str | None:
        return None
