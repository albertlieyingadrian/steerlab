"""The training judges: one holistic judge (single-judge arm) and a jury of four perspective-prompted jurors (jury arm).

This is the one file the judge-tuning loop edits (results/autoresearch/results.tsv). Every judge is the same open
model on the same server, at temperature 0; the arms differ only in their prompts and in how scores are combined.

Jury reward = mean of the two highest juror scores out of four ("credit answers some jurors love"): a conversation
one stakeholder rates highly is not averaged away by the others.
"""

import re

VERSION = "v1"

SINGLE = """You are evaluating a customer-service conversation between an AI agent and a customer. The agent works under the company policy below and can call tools on the company's systems.

Rate the agent's overall performance in this conversation from 1 (very poor) to 10 (excellent).

<policy>
{policy}
</policy>

<conversation>
{transcript}
</conversation>

Write at most three short sentences of reasoning, then a final line of the form "SCORE: N"."""

JURORS = {
    "customer": """You are the customer in the conversation below, looking back on how the company's AI agent handled your request. You care about whether you got what you came for, how much effort it took, and whether you were treated well.""",
    "compliance": """You are the company's policy compliance officer. You care about whether the agent followed every rule in the policy: verifying identity, getting explicit confirmation before changing anything, refusing what the policy forbids, and never inventing information.""",
    "operations": """You are the operations lead who reviews agents' tool use. You care about whether the agent called the right tools with the right arguments, read the results correctly, and finished the job accurately without wasted or wrong actions.""",
    "communication": """You are the head of customer communication. You care about whether the agent explained clearly what it found and what it did, set honest expectations, and kept the customer informed at each step.""",
}

JUROR_TEMPLATE = """{persona}

Rate the agent from your point of view from 1 (very poor) to 10 (excellent).

<policy>
{policy}
</policy>

<conversation>
{transcript}
</conversation>

Write at most three short sentences of reasoning, then a final line of the form "SCORE: N"."""

SCORE_RE = re.compile(r"SCORE:\s*(10|[1-9])\b")


def render_transcript(conv, max_tool_chars=700):
    """conv["transcript"]: list of {"role": "user"|"agent"|"tool_call"|"tool_result", "content": str}."""
    lines = []
    for m in conv["transcript"]:
        c = m["content"]
        if m["role"] == "tool_result" and len(c) > max_tool_chars:
            c = c[:max_tool_chars] + " ...[truncated]"
        tag = {"user": "Customer", "agent": "Agent", "tool_call": "Agent calls tool", "tool_result": "Tool result"}[
            m["role"]
        ]
        lines.append(f"[{tag}] {c}")
    end = conv.get("termination", "")
    if end and end != "user_stop":
        lines.append(f"[System] The conversation was cut off ({end}).")
    return "\n".join(lines)


def prompts_for(arm, conv, policy):
    t = render_transcript(conv)
    if arm == "single":
        return {"single": SINGLE.format(policy=policy, transcript=t)}
    return {k: JUROR_TEMPLATE.format(persona=p, policy=policy, transcript=t) for k, p in JURORS.items()}


def parse_score(text):
    m = SCORE_RE.findall(text or "")
    return int(m[-1]) if m else None


def aggregate(arm, scores):
    """scores: {judge_name: int or None}. Unparseable scores count as 1 (the floor), and are logged by the caller."""
    vals = [s if s is not None else 1 for s in scores.values()]
    if arm == "single":
        return float(vals[0])
    top = sorted(vals, reverse=True)[:2]
    return sum(top) / len(top)
