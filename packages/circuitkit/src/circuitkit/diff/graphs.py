"""A small, position-free summary of one attribution graph, and loaders for the formats graphs come in.

An attribution graph from circuit-tracer has one node per (layer, position, feature). Comparing two models' graphs
on the same prompt needs units that exist in both, so a summary sums each feature's influence over positions:

- `features`: `"layer:feature"` -> influence on the target, summed over the positions where it was active;
- `edges`: `("layer:feature", "layer:feature")` -> summed weight of the pruned graph's feature-to-feature edges;
- the graph's replacement and completeness scores and the influence that flows through error nodes, which say how
  much of the model the transcoders explain on this prompt.

Loaders: `from_circuit_tracer` (a `circuit_tracer.graph.Graph` or a `.pt` file), `from_neuronpedia` (the graph JSON
the Neuronpedia frontend reads) and `load_summaries` / `save_summaries` for this package's own JSON (optionally
gzipped).
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

FORMAT = "circuitkit.graphs/1"
Unit = str  # "layer:feature"


def unit(layer: int, feature: int) -> Unit:
    return f"{int(layer)}:{int(feature)}"


def parse_unit(u: Unit) -> tuple[int, int]:
    layer, feature = u.split(":")
    return int(layer), int(feature)


class GraphSummary(BaseModel):
    """One graph, reduced to per-unit influence and unit-to-unit edge weights."""

    name: str
    model: str | None = None
    prompt: str | None = None
    features: dict[Unit, float] = Field(description="unit -> influence on the target, summed over positions")
    edges: dict[tuple[Unit, Unit], float] = Field(
        default_factory=dict, description="(source unit, target unit) -> summed weight of the pruned graph's edges"
    )
    replacement_score: float | None = None
    completeness_score: float | None = None
    error_influence: float | None = Field(None, description="influence that flows through error nodes")
    meta: dict[str, Any] = Field(default_factory=dict)

    def present(self) -> set[Unit]:
        return set(self.features)

    def to_json(self) -> dict[str, Any]:
        d = self.model_dump(exclude={"edges"})
        d["edges"] = [[u, v, w] for (u, v), w in self.edges.items()]
        return d

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> GraphSummary:
        data = dict(d)
        data["edges"] = {(u, v): float(w) for u, v, w in data.get("edges", [])}
        return cls.model_validate(data)


def save_summaries(path: str | Path, graphs: Mapping[str, Sequence[GraphSummary]]) -> None:
    """Write named groups of summaries (e.g. {"before": [...], "after": [...]}) to JSON; `.gz` paths are gzipped."""
    payload = {"format": FORMAT, "groups": {k: [g.to_json() for g in v] for k, v in graphs.items()}}
    text = json.dumps(payload, separators=(",", ":"))
    p = Path(path)
    if p.suffix == ".gz":
        p.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
    else:
        p.write_text(text, encoding="utf-8")


def load_summaries(path: str | Path) -> dict[str, list[GraphSummary]]:
    p = Path(path)
    raw = gzip.decompress(p.read_bytes()).decode("utf-8") if p.suffix == ".gz" else p.read_text(encoding="utf-8")
    payload = json.loads(raw)
    if payload.get("format") != FORMAT:
        raise ValueError(f"{p} is not a {FORMAT} file (format={payload.get('format')!r})")
    return {k: [GraphSummary.from_json(g) for g in v] for k, v in payload["groups"].items()}


def _aggregate(
    feats: Sequence[tuple[int, int, int]],
    influence: Iterable[float],
    edges: Iterable[tuple[int, int, float]] = (),
    max_edges: int | None = None,
) -> tuple[dict[Unit, float], dict[tuple[Unit, Unit], float]]:
    """Sum node influence by unit; sum edges (target index, source index, weight) by unit pair, self-pairs dropped."""
    agg: dict[Unit, float] = {}
    for (layer, _pos, f), v in zip(feats, influence, strict=True):
        u = unit(layer, f)
        agg[u] = agg.get(u, 0.0) + float(v)
    eagg: dict[tuple[Unit, Unit], float] = {}
    for t, s, w in edges:
        a, b = feats[s], feats[t]
        su, tu = unit(a[0], a[2]), unit(b[0], b[2])
        if su != tu:
            eagg[(su, tu)] = eagg.get((su, tu), 0.0) + float(w)
    if max_edges is not None and len(eagg) > max_edges:
        eagg = dict(sorted(eagg.items(), key=lambda kv: -abs(kv[1]))[:max_edges])
    return agg, eagg


def from_feature_lists(
    name: str,
    features: Sequence[Sequence[int]],
    influence: Sequence[float],
    edges: Sequence[Sequence[float]] = (),
    max_edges: int | None = None,
    **kw: Any,
) -> GraphSummary:
    """Build a summary from per-node lists: `features[i] = (layer, pos, feature)`, `influence[i]`, and edges as
    `(target node index, source node index, weight)` over the same indices (circuit-tracer's adjacency order)."""
    feats = [(int(a[0]), int(a[1]), int(a[2])) for a in features]
    e = [(int(x[0]), int(x[1]), float(x[2])) for x in edges]
    f, ed = _aggregate(feats, influence, e, max_edges)
    return GraphSummary(name=name, features=f, edges=ed, **kw)


def from_circuit_tracer(
    graph: Any,
    name: str = "graph",
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    max_edges: int | None = 5000,
) -> GraphSummary:
    """Summarise a `circuit_tracer.graph.Graph` (or a path to a saved `.pt` graph).

    Influence is circuit-tracer's `compute_node_influence` weighted by the logit targets' probabilities; edges are
    the feature-to-feature edges kept by `prune_graph(node_threshold, edge_threshold)`. Needs circuit-tracer and torch.
    """
    import torch
    from circuit_tracer.graph import Graph, compute_graph_scores, compute_node_influence, prune_graph

    if isinstance(graph, str | Path):
        graph = Graph.from_pt(str(graph))
    rep, comp = compute_graph_scores(graph)
    adj = graph.adjacency_matrix
    n = adj.shape[0]
    lw = torch.zeros(n, device=adj.device)
    probs = graph.logit_probabilities.to(adj.device)
    lw[-len(probs) :] = probs
    infl = compute_node_influence(adj, lw)
    nf = len(graph.selected_features)
    feats = [tuple(x) for x in graph.active_features[graph.selected_features].cpu().tolist()]
    pruned = prune_graph(graph, node_threshold=node_threshold, edge_threshold=edge_threshold)
    em = pruned.edge_mask[:nf, :nf].nonzero().cpu()
    ew = adj[:nf, :nf].cpu()[em[:, 0], em[:, 1]].float().tolist()
    n_tok = len(graph.input_tokens)
    n_layers = int(graph.cfg.n_layers)
    e0, e1 = nf, nf + n_tok * n_layers
    edges = [(int(a), int(b), float(w)) for (a, b), w in zip(em.tolist(), ew, strict=True)]
    f, ed = _aggregate(feats, infl[:nf].float().cpu().tolist(), edges, max_edges)
    return GraphSummary(
        name=name,
        model=str(getattr(graph.cfg, "model_name", "") or "") or None,
        features=f,
        edges=ed,
        replacement_score=float(rep),
        completeness_score=float(comp),
        error_influence=float(infl[e0:e1].sum()),
        meta={"n_tokens": n_tok, "n_features": nf},
    )


def from_neuronpedia(data: Mapping[str, Any] | str | Path, name: str | None = None) -> GraphSummary:
    """Summarise a Neuronpedia / circuit-tracer frontend graph JSON (`nodes` + `links`).

    Feature nodes have ids `"{layer}_{feature}_{pos}"` (their `feature` field is a Cantor pairing of layer and
    index, so the id is parsed instead); error, embedding and logit nodes are skipped. A node's `influence` in these
    files is cumulative (the share of all influence held by the nodes ranked at or above it), so per-node influence
    is recovered as the step from the previous node, in rank order, over every node in the file. Nodes the pruning
    dropped make the step after them a little too big, so treat these values as approximate shares.
    """
    if isinstance(data, str | Path):
        data = json.loads(Path(data).read_text(encoding="utf-8"))
    assert isinstance(data, Mapping)
    nodes = data["nodes"]
    ranked = sorted((nd for nd in nodes if nd.get("influence") is not None), key=lambda nd: float(nd["influence"]))
    per_node: dict[str, float] = {}
    prev = 0.0
    for nd in ranked:
        per_node[nd["node_id"]] = float(nd["influence"]) - prev
        prev = float(nd["influence"])
    feat_nodes = [nd for nd in nodes if str(nd.get("feature_type", "")).endswith("transcoder")]
    feats: dict[Unit, float] = {}
    id_unit: dict[str, Unit] = {}
    for nd in feat_nodes:
        layer, feature, _pos = str(nd["node_id"]).split("_")
        u = unit(int(layer), int(feature))
        id_unit[nd["node_id"]] = u
        feats[u] = feats.get(u, 0.0) + per_node.get(nd["node_id"], 0.0)
    edges: dict[tuple[Unit, Unit], float] = {}
    for link in data.get("links", []):
        s, t = id_unit.get(link["source"]), id_unit.get(link["target"])
        if s and t and s != t:
            edges[(s, t)] = edges.get((s, t), 0.0) + float(link["weight"])
    meta = data.get("metadata", {})
    return GraphSummary(
        name=name or str(meta.get("slug", "graph")),
        model=meta.get("scan"),
        prompt=meta.get("prompt"),
        features=feats,
        edges=edges,
        meta={"source": "neuronpedia"},
    )
