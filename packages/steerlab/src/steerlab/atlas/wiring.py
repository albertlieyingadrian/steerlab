"""Wiring: what drives a unit and what it drives, walked layer by layer into a small graph.

    graph = walk(edges, title, Address.parse("L8/n344+"), up=2, down=2, fan=4)
    graph["nodes"]   # [{"address", "layer", "title", "hop"}], hop < 0 upstream, 0 the unit, > 0 downstream
    graph["links"]   # [{"source", "target", "push", "cos"}], source always the earlier layer

`edges(address)` returns `{"in": [...], "out": [...]}` lists of `{"address", "push", "cos"}` (the catalog's `edges`
table, or `compute.unit_edges`). Each hop keeps the `fan` strongest links by |push|, so the graph stays readable:
`fan` 4 and 2 hops each way is at most 1 + 4 + 16 nodes a side. Edges are wiring (weights), not traffic: a strong edge
says the earlier pole's write points along the later pole's read direction, not that it fires in a given text.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from steerlab.atlas.address import Address

EdgesFn = Callable[[Address], dict[str, Any]]
TitleFn = Callable[[Address], str]


def _strength(edge: dict[str, Any]) -> float:
    value = edge.get("push")
    if value is None:
        value = edge.get("cos")
    return abs(float(value or 0.0))


def walk(edges: EdgesFn, title: TitleFn, center: Address, up: int = 2, down: int = 2, fan: int = 4) -> dict[str, Any]:
    nodes: dict[Address, dict[str, Any]] = {center: {"address": center.path(), "layer": center.layer, "hop": 0}}
    links: dict[tuple[str, str], dict[str, Any]] = {}
    cache: dict[Address, dict[str, Any]] = {}

    def edges_of(address: Address) -> dict[str, Any]:
        if address not in cache:
            try:
                cache[address] = edges(address)
            except (FileNotFoundError, ValueError):
                cache[address] = {"in": [], "out": []}
        return cache[address]

    for side, hops, direction in (("in", up, -1), ("out", down, 1)):
        frontier = [center]
        for hop in range(1, hops + 1):
            nxt: list[Address] = []
            for address in frontier:
                for item in sorted(edges_of(address).get(side, []), key=_strength, reverse=True)[:fan]:
                    other = Address.parse(item["address"])
                    source, target = (other, address) if side == "in" else (address, other)
                    key = (source.path(), target.path())
                    if key not in links:
                        links[key] = {
                            "source": key[0],
                            "target": key[1],
                            "push": item.get("push"),
                            "cos": item.get("cos"),
                        }
                    if other not in nodes:
                        nodes[other] = {"address": other.path(), "layer": other.layer, "hop": direction * hop}
                        nxt.append(other)
            frontier = nxt
    for address, node in nodes.items():
        node["display"] = str(address)
        node["title"] = title(address)
    ordered = sorted(nodes.values(), key=lambda n: (n["layer"], n["hop"], n["address"]))
    return {
        "center": center.path(),
        "nodes": ordered,
        "links": list(links.values()),
        "how": "direct weight edges, strongest first by |push|; wiring, not traffic",
    }
