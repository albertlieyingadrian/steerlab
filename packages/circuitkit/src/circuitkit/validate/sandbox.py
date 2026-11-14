"""Compile explanation rules, `predict(tokens, i) -> float`, in a restricted namespace.

A rule is a feature's explanation written as code: given the prompt's tokens and a position, it predicts the
feature's activation there. Gao's swap test replaces the feature with this prediction and asks how much of the
feature's effect survives.

The namespace is restricted to catch honest mistakes, not to contain hostile code: rules written by a language
model are still code, so only run rules you have read or generated yourself.

Two lessons from the circuits notes are built in: every name a rule uses is checked before it runs (a missing
builtin otherwise raises inside the rule's own `try/except` and the rule silently returns 0 everywhere), and `id`
hashes the token list instead of returning an address that a later list can reuse.
"""

from __future__ import annotations

import ast
import builtins
import math
from collections.abc import Callable, Sequence

Rule = Callable[[list[str], int], float]

ALLOWED_BUILTINS = [
    "abs",
    "all",
    "any",
    "bool",
    "chr",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "frozenset",
    "int",
    "isinstance",
    "len",
    "list",
    "map",
    "max",
    "min",
    "ord",
    "pow",
    "range",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
    "Exception",
    "ArithmeticError",
    "IndexError",
    "KeyError",
    "TypeError",
    "ValueError",
    "ZeroDivisionError",
]


class RuleError(ValueError):
    """The rule cannot be compiled or uses a name the namespace does not provide."""


def _namespace() -> dict[str, object]:
    safe: dict[str, object] = {n: getattr(builtins, n) for n in ALLOWED_BUILTINS}
    safe["id"] = lambda o: hash(tuple(o)) if isinstance(o, list) else builtins.id(o)
    return {"__builtins__": safe, "math": math}


def unknown_names(code: str) -> set[str]:
    """Names the rule reads that it neither defines nor gets from the namespace."""
    tree = ast.parse(code)
    defined: set[str] = set()
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (defined if isinstance(node.ctx, ast.Store | ast.Del) else used).add(node.id)
        elif isinstance(node, ast.FunctionDef | ast.Lambda):
            if isinstance(node, ast.FunctionDef):
                defined.add(node.name)
            args = node.args
            for a in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
                if a is not None:
                    defined.add(a.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
    available = set(ALLOWED_BUILTINS) | {"id", "math", "True", "False", "None"}
    return used - defined - available


def compile_rule(code: str) -> Rule:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise RuleError(f"rule does not parse: {e}") from e
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            raise RuleError("rules may not import modules")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise RuleError(f"rules may not use dunder attributes ({node.attr})")
    missing = unknown_names(code)
    if missing:
        raise RuleError(f"rule uses names the namespace does not provide: {sorted(missing)}")
    ns = _namespace()
    exec(compile(tree, "<rule>", "exec"), ns)
    fn = ns.get("predict")
    if not callable(fn):
        raise RuleError("rule must define predict(tokens, i)")
    return fn


def rule_values(fn: Rule, tokens: Sequence[str]) -> tuple[list[float], int]:
    """The rule's prediction at every position; a call that raises or returns a non-number counts as 0 and an error."""
    vals: list[float] = []
    errors = 0
    for i in range(len(tokens)):
        try:
            v = float(fn(list(tokens), i))
            if not math.isfinite(v):
                raise ValueError("not finite")
        except Exception:
            v, errors = 0.0, errors + 1
        vals.append(v)
    return vals, errors
