"""The prompts for every arm. Frozen after dev tuning (see results.tsv); held-out runs use exactly these.

Baggage arms:
  a   no rules: the question and RuleArena's instruction, from what the model already knows;
  b0  RuleArena's own prompt: the slice's rules pasted (with their source line numbers), free-form answer;
  b   the same, plus a JSON trace (facts, and for each checked bag the line used and the fee for each component).
      Arm c is not a prompt: it is the checker run on b's traces (score.py).
Contrast arms: a (no rules) and b (rules pasted, with a JSON trace).
"""

from __future__ import annotations

from common import SPEC_DIR

PROMPT_VERSION = "v1"
RULES = (SPEC_DIR / "baggage_rules_slice.txt").read_text(encoding="utf-8")
DESIGN_RULES = (SPEC_DIR / "contrast_rules.txt").read_text(encoding="utf-8")

SYSTEM = (
    "You are a helpful assistant at an airline. You are given the information of a passenger, his / her items, his / "
    "her special needs, and the policies of the airline. You should compute the total cost (including the flight "
    "ticket fee, checked bag fees, cost of special needs) according to the policies for the passenger."
)  # RuleArena's system prompt with the airline's name replaced
SYSTEM_NO_RULES = (
    "You are a helpful assistant at a major U.S. airline. You are given the information of a passenger and his / her "
    "items. You should compute the total cost (including the flight ticket fee and checked bag fees) according to the "
    "airline's 2024 checked-bag policy, as you know it."
)
ASK = (
    "Compute the total cost for him step by step (don't omit any bag) and end your response with \"The total cost is "
    '$xxx." (xxx is a number)'
)
TRACE = """After your step-by-step calculation, write a trace of it as one JSON object in a ```json block, citing the \
line labels (L...) of the rules you used, in this shape:
{"class": "<fare class>", "region": "<one of: U.S., Mexico, Cuba, Europe, Japan, Australia>",
 "direction": "<from_us or to_us>", "ticket": <number>,
 "checked_bags": [{"item": <item number in the list>, "position": <1 for the bag charged as the first checked bag, 2, 3>,
   "size": <length + width + height>, "weight": <lbs>,
   "base_line": "L..", "base_fee": <number>, "weight_line": "L..", "weight_fee": <number>,
   "size_line": "L..", "size_fee": <number>, "bag_fee": <number>}],
 "total": <number>}
List only checked bags (not the carry-on). For a bag within the weight or size allowance, cite the allowance line \
and a fee of 0. Then end your response with "The total cost is $xxx." (xxx is a number)"""


def baggage_messages(arm: str, prompt: str) -> list[dict]:
    if arm == "a":
        return [{"role": "system", "content": SYSTEM_NO_RULES}, {"role": "user", "content": f"{prompt} {ASK}"}]
    head = f"The policies of the airline are as follows (each line starts with its label):\n\n{RULES}\n"
    if arm == "b0":
        user = f"{head}{prompt} {ASK}\nYour response:"
    elif arm == "b":
        user = f"{head}{prompt} Compute the total cost for him step by step (don't omit any bag).\n\n{TRACE}"
    else:
        raise ValueError(arm)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


DESIGN_SYSTEM = (
    "You are a design-system reviewer. You check whether an interface element follows a design system's rules."
)
DESIGN_ASK = (
    "Does this element follow the design system's colour rules (including colour contrast)? Work it out step by step, "
    'then end your response with "Verdict: compliant" or "Verdict: not compliant".'
)
DESIGN_TRACE = """Does this element follow the rules below? Work it out step by step. Then write a trace as one JSON \
object in a ```json block, citing the line labels (D...) of the rules, in this shape:
{"facts": {"element": "<as named in the question>", "size_px": <number>, "bold": <true or false>,
   "focused": <true or false>, "text_colour": "<hex>", "background": "<hex>"},
 "contrast_rule": "<the one line that sets this element's contrast requirement: D31, D32, D33 or D34>",
 "ratio": <contrast ratio, 2 decimals>, "rules_failed": ["<label of each rule the element breaks>"],
 "verdict": "<compliant or not compliant>"}
Then end your response with "Verdict: compliant" or "Verdict: not compliant"."""


def contrast_messages(arm: str, prompt: str) -> list[dict]:
    if arm == "a":
        return [{"role": "system", "content": DESIGN_SYSTEM}, {"role": "user", "content": f"{prompt} {DESIGN_ASK}"}]
    if arm == "b":
        user = f"The design system's rules (each line starts with its label):\n\n{DESIGN_RULES}\n{prompt}\n\n{DESIGN_TRACE}"
        return [{"role": "system", "content": DESIGN_SYSTEM}, {"role": "user", "content": user}]
    raise ValueError(arm)


def messages(task: str, arm: str, prompt: str) -> list[dict]:
    return baggage_messages(arm, prompt) if task == "baggage" else contrast_messages(arm, prompt)
