"""Guard levels. Each is a list of guard modules, so any one can be switched on or off by editing a list."""

from __future__ import annotations

from .guards.budgets import Budgets
from .guards.egress_dlp import EgressDLP
from .guards.egress_dlp_plus import EgressDLPPlus
from .guards.keyword import KeywordFilter
from .guards.provenance import Provenance
from .guards.quarantine import Quarantine
from .guards.regex_dlp import RegexDLP
from .guards.sql_firewall import SQLFirewall

L1 = [KeywordFilter]
L2 = L1 + [RegexDLP]
L3 = L2 + [SQLFirewall, Provenance, Budgets, EgressDLP]

LEVELS = {
    "L0": [],
    "L1": L1,
    "L2": L2,
    "L3": L3,
    "L3-kw": [g for g in L3 if g is not KeywordFilter],  # ablation: L3 without the L1 keyword filter
    "L2+P1": L2 + [Quarantine],
    "L3+P1": [Quarantine] + L3,
    "L3+P2": [g for g in L3 if g is not EgressDLP] + [EgressDLPPlus],
    "L3+P1+P2": [Quarantine] + [g for g in L3 if g is not EgressDLP] + [EgressDLPPlus],
}
# The system prompt carries data-protection rules from L1 up; L0 is the bare agent.
PROMPT_RULES = {k: k != "L0" for k in LEVELS}


def build(level: str) -> list:
    return [g() for g in LEVELS[level]]
