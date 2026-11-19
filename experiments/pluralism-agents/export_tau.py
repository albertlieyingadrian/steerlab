"""Export the tau-bench (tau2-bench repo, v2.0.0) retail and airline tasks, policies, tool schemas and the text-mode
user-simulator system prompt to data/tau_tasks.json, so the Modal job reads one frozen file.

Run with a venv that has tau2 installed (pinned commit in README):
    python experiments/pluralism-agents/export_tau.py
"""

import json
from pathlib import Path

from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
from tau2.data_model.persona import PersonaConfig
from tau2.registry import registry
from tau2.user.guidelines import simulation_guidelines
from tau2.user.user_simulator import build_user_system_prompt

OUT = Path(__file__).resolve().parent / "data" / "tau_tasks.json"
WRITE_TOOLS = {
    "airline": [
        "book_reservation",
        "cancel_reservation",
        "update_reservation_flights",
        "update_reservation_baggages",
        "update_reservation_passengers",
        "send_certificate",
    ],
    "retail": [
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
        "return_delivered_order_items",
    ],
}


def main():
    out = {}
    persona = PersonaConfig()
    guidelines = simulation_guidelines(persona, mode="text", use_tools=False)
    for domain in ["retail", "airline"]:
        env = registry.get_env_constructor(domain)()
        tasks_fn = registry.get_tasks_loader(domain)
        splits = {s: [t.id for t in tasks_fn(s)] for s in ["train", "test"]}
        tasks = tasks_fn(None)
        tools = [t.openai_schema for t in env.get_tools()]
        mut = {t.name: env._is_mutating_tool(t.name) for t in env.get_tools()}
        assert {n for n, m in mut.items() if m} == set(WRITE_TOOLS[domain]), mut
        rows = []
        for t in tasks:
            ec = t.evaluation_criteria
            rows.append(
                {
                    "id": t.id,
                    "user_system": build_user_system_prompt(
                        str(t.user_scenario), persona_config=persona, mode="text", guidelines=guidelines
                    ),
                    "actions": [{"name": a.name, "arguments": a.arguments} for a in (ec.actions or [])],
                    "communicate_info": ec.communicate_info or [],
                    "reward_basis": [str(r.value) for r in ec.reward_basis],
                    "nl_assertions": ec.nl_assertions or [],
                    "purpose": t.description.purpose if t.description else None,
                    "split": next((s for s, ids in splits.items() if t.id in ids), None),
                }
            )
        out[domain] = {
            "agent_system": SYSTEM_PROMPT.format(domain_policy=env.get_policy(), agent_instruction=AGENT_INSTRUCTION),
            "tools": tools,
            "write_tools": WRITE_TOOLS[domain],
            "tasks": rows,
        }
        print(domain, len(rows), {s: len(v) for s, v in splits.items()}, len(tools), "tools")
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
