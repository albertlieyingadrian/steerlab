"""The tau-bench environment as a small JSON-lines server, run in its own Python (the tau2 package and its
dependencies live in a separate venv so they cannot clash with vLLM's).

Protocol: one JSON request per line on stdin, one JSON reply per line on stdout.
    {"calls": [{"op": "new", "key": "c1", "domain": "airline"},
               {"op": "call", "key": "c1", "name": "get_user_details", "arguments": {...}},
               {"op": "score", "key": "c1", "task": {...}, "agent_texts": [...]},
               {"op": "close", "key": "c1"}]}
    -> {"results": [...]}   (one result per call, in order)

`score` mirrors tau2's evaluator for the half-duplex text mode: DB reward = the environment after the agent's tool
calls has the same hash as a fresh environment after the task's gold actions; COMMUNICATE (airline) = every
communicate_info string appears in some agent text, lower-cased with commas removed. NL assertions (in 40 retail
tasks' reward basis) need an LLM and are not scored here; the README says so.
"""

import json
import sys

from loguru import logger
from tau2.data_model.message import ToolCall
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.environment import get_environment as airline_env
from tau2.domains.airline.utils import AIRLINE_DB_PATH
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.environment import get_environment as retail_env
from tau2.domains.retail.utils import RETAIL_DB_PATH

logger.remove()  # tau2 logs every tool response at DEBUG to stderr; keep the container logs readable

BASE_DB = {"airline": FlightDB.load(AIRLINE_DB_PATH), "retail": RetailDB.load(RETAIL_DB_PATH)}
MAKE = {"airline": airline_env, "retail": retail_env}
ENVS = {}
GOLD = {}


def new_env(domain):
    return MAKE[domain](db=BASE_DB[domain])


def gold_hash(domain, task):
    key = (domain, task["id"])
    if key not in GOLD:
        env = new_env(domain)
        for a in task["actions"]:
            try:
                env.make_tool_call(tool_name=a["name"], requestor="assistant", **a["arguments"])
            except Exception as e:  # tau2 logs and continues on a failing gold action; so do we
                print(f"gold action failed {key} {a['name']}: {e}", file=sys.stderr)
        GOLD[key] = env.get_db_hash()
    return GOLD[key]


def handle(c):
    op = c["op"]
    if op == "new":
        ENVS[c["key"]] = (c["domain"], new_env(c["domain"]))
        return {"ok": True}
    if op == "call":
        _, env = ENVS[c["key"]]
        args = c["arguments"] if isinstance(c["arguments"], dict) else {}
        msg = env.get_response(ToolCall(id="t", name=c["name"], arguments=args, requestor="assistant"))
        return {"content": msg.content, "error": bool(msg.error)}
    if op == "score":
        domain, env = ENVS[c["key"]]
        task = c["task"]
        db = float(env.get_db_hash() == gold_hash(domain, task))
        reward = db
        comm = 1.0
        if "COMMUNICATE" in task["reward_basis"] and task["communicate_info"]:
            texts = [t.lower().replace(",", "") for t in c["agent_texts"]]
            comm = float(all(any(s.lower() in t for t in texts) for s in task["communicate_info"]))
            reward *= comm
        return {"reward": reward, "db": db, "communicate": comm}
    if op == "close":
        ENVS.pop(c["key"], None)
        return {"ok": True}
    raise ValueError(op)


def main():
    for line in sys.stdin:
        req = json.loads(line)
        out = []
        for c in req["calls"]:
            try:
                out.append(handle(c))
            except Exception as e:
                out.append({"error_msg": f"{type(e).__name__}: {e}"})
        sys.stdout.write(json.dumps({"results": out}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
