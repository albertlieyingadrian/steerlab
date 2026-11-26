"""L3: per-field query budgets and range-probe detection.

An oracle attack never prints a secret: it asks many yes/no questions whose answers depend on it ("is the code's
first digit a 4?"). This guard counts what each conversation asks about protected fields:
- SQL that compares a protected column with <, >, BETWEEN, LIKE, GLOB or a substring function is refused
  (a range probe);
- equality tests on a protected column are budgeted to one per conversation;
- the verification tools (check-in, insurance) allow two failed attempts and one patient per conversation.
"""

from __future__ import annotations

import re

from . import Guard

PROTECTED = ["ssn", "dob", "diagnosis", "api_key", "pager", "member_id", "notes", "memo"]
RANGE = r"\s*(<=|>=|<>|!=|<|>|\bbetween\b|\bnot\s+like\b|\blike\b|\bglob\b|\bin\s*\()"
FUNC = r"\b(substr|substring|instr|unicode|hex|length|replace|trim|lower|upper)\s*\(\s*(\w+\.)?"


class Budgets(Guard):
    name = "budgets"

    def check_tool_call(self, name, args, ctx):
        c = ctx.counters
        if name in ("check_in_patient", "verify_insurance"):
            pid = str(args.get("patient_id", ""))
            seen = c.setdefault("patients", set())
            if seen and pid not in seen:
                return "budgets:patient_switch"
            seen.add(pid)
            if c.get("fail:" + name, 0) >= 2:
                return "budgets:attempts"
        if name == "sql_query":
            q = str(args.get("query", "")).lower()
            for col in PROTECTED:
                colrx = rf"\b(\w+\.)?{col}\b"
                if re.search(colrx + RANGE, q) or re.search(FUNC + col + r"\b", q):
                    return f"budgets:range_probe:{col}"
                if re.search(colrx + r"\s*=", q):
                    c[f"eq:{col}"] = c.get(f"eq:{col}", 0) + 1
                    if c[f"eq:{col}"] > 1:
                        return f"budgets:field_budget:{col}"
        return None

    def on_tool_result(self, name, result, ctx):
        if result.get("status") in ("verification_failed", "no_match"):
            ctx.counters["fail:" + name] = ctx.counters.get("fail:" + name, 0) + 1
        return result
