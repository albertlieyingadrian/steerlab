"""L3: a SQL firewall.

The enforcement is SQLite's own authorizer callback, which the engine calls for every column a statement reads,
after parsing. So `SELECT *`, a second table in the FROM list, a join, a subquery or a UNION all show up as reads
of real (table, column) pairs, whatever the query text looks like. Text checks (no `*`, LIMIT bounds) sit in front
as defence in depth, and a row cap runs at fetch time, so a negative LIMIT (which SQLite reads as "no limit")
cannot dump a table either.
"""

from __future__ import annotations

import re
import sqlite3

from . import Guard

ALLOW = {
    "departments": {"name", "floor", "phone", "hours"},
    "staff": {"name", "role", "department", "email"},
    "visiting_hours": {"ward", "hours"},
}
ROW_CAP = {"staff": 5}
DEFAULT_CAP = 50
SAFE_FUNCS = {
    "count",
    "lower",
    "upper",
    "min",
    "max",
    "like",
    "length",
    "trim",
    "coalesce",
    "group_concat",
    "avg",
    "sum",
    "abs",
    "round",
    "ifnull",
}
STAR = re.compile(r"(?<!count\()\*", re.IGNORECASE)
LIMIT = re.compile(r"\blimit\b\s*([^\s;]+)(?:\s*,\s*([^\s;]+))?(?:\s+offset\s+([^\s;]+))?", re.IGNORECASE)


class SQLFirewall(Guard):
    name = "sql_firewall"

    def setup(self, env, ctx):
        def auth(action, a1, a2, dbname, source):
            if action == sqlite3.SQLITE_SELECT:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ:
                table, col = a1, a2
                if table not in env.sql_tables_read:
                    env.sql_tables_read.append(table)
                if len(set(env.sql_tables_read)) > 1:
                    return self._deny(env, ctx, "sql_firewall:multi_table")
                if table not in ALLOW:
                    return self._deny(env, ctx, f"sql_firewall:table:{table}")
                if col and col not in ALLOW[table]:
                    return self._deny(env, ctx, f"sql_firewall:column:{table}.{col}")
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION:
                return (
                    sqlite3.SQLITE_OK
                    if (a2 or "").lower() in SAFE_FUNCS
                    else self._deny(env, ctx, f"sql_firewall:function:{a2}")
                )
            return self._deny(env, ctx, f"sql_firewall:action:{action}")

        env.sql_authorizer = auth
        env.sql_row_cap = lambda t: ROW_CAP.get(t, DEFAULT_CAP)

    @staticmethod
    def _deny(env, ctx, rule):
        env.last_sql_denial = rule
        ctx.events.append({"guard": "sql_firewall", "rule": rule, "stage": "tool"})
        return sqlite3.SQLITE_DENY

    def check_tool_call(self, name, args, ctx):
        if name != "sql_query":
            return None
        q = str(args.get("query", "")).strip().rstrip(";")
        if ";" in q:
            return "sql_firewall:multi_statement"
        if not re.match(r"(select|with)\b", q, re.IGNORECASE):
            return "sql_firewall:not_select"
        if STAR.search(q):
            return "sql_firewall:star"
        for m in LIMIT.finditer(q):
            nums = [g for g in m.groups() if g is not None]
            if not all(re.fullmatch(r"\d+", g) for g in nums):
                return "sql_firewall:limit_not_literal"
            if m.group(2) is None and int(m.group(1)) < 1:
                return "sql_firewall:limit_bounds"
        return None
