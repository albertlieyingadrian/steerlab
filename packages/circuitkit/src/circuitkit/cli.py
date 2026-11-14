"""The `circuitkit` command: `diff`, `summarize`, `validate`, `label` and `version`.

circuitkit diff graphs.json.gz --before installed/before --after installed/after --html diff.html
circuitkit summarize graph_*.pt --group after --out graphs.json.gz
circuitkit validate --demo --html validate.html
circuitkit label --neuronpedia gemma-2-2b/20-gemmascope-transcoder-16k --features 15589,4012 --html labels.html
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from circuitkit import __version__

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Diff, validate and label circuit-tracer circuits.")


def _write(obj_json: str, html_text: str, json_path: Path | None, html_path: Path | None) -> None:
    from circuitkit.report import save_html

    if json_path:
        json_path.write_text(obj_json, encoding="utf-8")
        typer.echo(f"wrote {json_path}")
    if html_path:
        save_html(html_text, html_path)
        typer.echo(f"wrote {html_path}")


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@app.command()
def summarize(
    graphs: Annotated[list[Path], typer.Argument(help="circuit-tracer .pt graphs or Neuronpedia graph .json files")],
    out: Annotated[Path, typer.Option(help="summary file to create or extend (.json or .json.gz)")],
    group: Annotated[str, typer.Option(help="group name for these graphs, e.g. 'before'")],
) -> None:
    """Turn graphs into the position-free summaries `diff` reads (one group per call, appended to OUT)."""
    from circuitkit.diff.graphs import (
        GraphSummary,
        from_circuit_tracer,
        from_neuronpedia,
        load_summaries,
        save_summaries,
    )

    groups: dict[str, list[GraphSummary]] = load_summaries(out) if out.exists() else {}
    items = []
    for g in graphs:
        items.append(from_circuit_tracer(g, name=g.stem) if g.suffix == ".pt" else from_neuronpedia(g, name=g.stem))
    groups[group] = items
    save_summaries(out, groups)
    typer.echo(f"{out}: group {group!r} has {len(items)} graphs")


@app.command()
def diff(
    file: Annotated[Path, typer.Argument(help="summary file from `summarize` (or the bundled demo data)")],
    before: Annotated[str, typer.Option(help="group of graphs before training")],
    after: Annotated[str, typer.Option(help="group of graphs after training, same prompts in the same order")],
    control_before: Annotated[
        str | None, typer.Option(help="control run (did not learn the behaviour), before")
    ] = None,
    control_after: Annotated[str | None, typer.Option(help="control run, after")] = None,
    replication_before: Annotated[str | None, typer.Option(help="another seed, before")] = None,
    replication_after: Annotated[str | None, typer.Option(help="another seed, after")] = None,
    top_k: int = 10,
    min_present: int = 6,
    seed: int = 0,
    html: Annotated[Path | None, typer.Option(help="write the HTML report here")] = None,
    json_out: Annotated[Path | None, typer.Option("--json", help="write the JSON report here")] = None,
) -> None:
    """Which features and edges did training add or remove? With a stability check, a null and validity checks."""
    from circuitkit.diff.core import Settings, diff_graphs
    from circuitkit.diff.graphs import load_summaries
    from circuitkit.report import render_diff

    g = load_summaries(file)
    fb = g[before][0].meta.get("fvu_by_layer")
    fa = g[after][0].meta.get("fvu_by_layer")
    rep = diff_graphs(
        g[before],
        g[after],
        settings=Settings(top_k=top_k, min_present=min_present, seed=seed),
        control=(g[control_before], g[control_after]) if control_before and control_after else None,
        replication=(g[replication_before], g[replication_after]) if replication_before and replication_after else None,
        fvu_before=fb,
        fvu_after=fa,
        before_name=before,
        after_name=after,
    )
    s = rep.summary
    typer.echo(f"gained {s.n_gained}, lost {s.n_lost}, up {s.n_up}, down {s.n_down}; total change p = {s.change_p:.3f}")
    for c in rep.top:
        typer.echo(
            f"  {c.unit:>12}  {c.status:7} Δ={c.delta:+.4f}  "
            f"graphs {c.present_before}->{c.present_after}  p={c.p_null:.3f}"
        )
    typer.echo("validity: " + ("ok" if rep.validity.ok else "; ".join(rep.validity.warnings)))
    _write(rep.model_dump_json(indent=1), render_diff(rep), json_out, html)


@app.command()
def validate(
    demo: Annotated[bool, typer.Option(help="run on the built-in toy model (no GPU, no downloads)")] = False,
    model: Annotated[str | None, typer.Option(help="HF model id for circuit-tracer")] = None,
    transcoders: Annotated[str | None, typer.Option(help="transcoder set (HF id or local path)")] = None,
    prompt: Annotated[str | None, typer.Option(help="the prompt")] = None,
    counterfactual: Annotated[
        str | None, typer.Option(help="a prompt with a different answer, for the patch test")
    ] = None,
    features: Annotated[str | None, typer.Option(help="comma-separated layer:feature@pos, e.g. 20:15589@9")] = None,
    target: Annotated[str | None, typer.Option(help="answer token, e.g. ' Austin'")] = None,
    contrast: Annotated[str | None, typer.Option(help="contrast token, e.g. ' Dallas'")] = None,
    n_controls: int = 20,
    steer_scale: float = 3.0,
    seed: int = 0,
    device: Annotated[str | None, typer.Option(help="cuda, mps or cpu")] = None,
    html: Annotated[Path | None, typer.Option(help="write the HTML report here")] = None,
    json_out: Annotated[Path | None, typer.Option("--json", help="write the JSON report here")] = None,
) -> None:
    """Is each group of features a switch or a readout? Ablation, steering and patching against controls."""
    from circuitkit.report import render_validate
    from circuitkit.validate import FeatureNode, logit_diff
    from circuitkit.validate import validate as run

    if demo:
        from circuitkit.validate.toy import demo as toy
        from circuitkit.validate.toy import demo_nodes

        b = toy()
        metric = logit_diff(b.token_id(" Austin"), b.token_id(" Sacramento"), " Austin", " Sacramento")
        rep = run(b, "dallas", demo_nodes(b), metric=metric, counterfactual="oakland", n_controls=n_controls, seed=seed)
    else:
        if not (model and transcoders and prompt and features and target and contrast):
            raise typer.BadParameter(
                "give --demo, or all of --model --transcoders --prompt --features --target --contrast"
            )
        import torch
        from circuit_tracer import ReplacementModel

        from circuitkit.tracer import TracerBackend

        dev = torch.device(device) if device else None
        rm = ReplacementModel.from_pretrained(model, transcoders, device=dev, dtype=torch.bfloat16)
        b2 = TracerBackend(rm)
        nodes = []
        for spec in features.split(","):
            lf, pos = spec.strip().split("@")
            layer, feat = lf.split(":")
            nodes.append(FeatureNode(layer=int(layer), pos=int(pos), feature=int(feat), activation=0.0))
        metric = logit_diff(b2.token_id(target), b2.token_id(contrast), target, contrast)
        rep = run(
            b2,
            prompt,
            nodes,
            metric=metric,
            counterfactual=counterfactual,
            n_controls=n_controls,
            steer_scale=steer_scale,
            seed=seed,
            progress=typer.echo,
        )
    for g in rep.groups:
        typer.echo(f"  {g.name:>20}  {g.verdict.value}")
    _write(rep.model_dump_json(indent=1), render_validate(rep), json_out, html)


@app.command()
def label(
    neuronpedia: Annotated[
        str | None, typer.Option(help="model/source on Neuronpedia, e.g. gemma-2-2b/20-gemmascope-transcoder-16k")
    ] = None,
    features: Annotated[str | None, typer.Option(help="comma-separated feature indices (with --neuronpedia)")] = None,
    records: Annotated[Path | None, typer.Option(help="JSON list of {unit, records:[{tokens, acts}]} instead")] = None,
    provider: Annotated[str, typer.Option(help="anthropic (Claude Haiku 5.5) or openai")] = "anthropic",
    model_name: Annotated[str | None, typer.Option("--model", help="override the provider's default model")] = None,
    score: Annotated[str, typer.Option(help="detection, simulation, both or none")] = "detection",
    compare_neuronpedia: Annotated[bool, typer.Option("--compare", help="also score Neuronpedia's labels")] = False,
    cache: Annotated[Path | None, typer.Option(help="LLM cache file (JSON lines)")] = None,
    html: Annotated[Path | None, typer.Option(help="write the HTML report here")] = None,
    json_out: Annotated[Path | None, typer.Option("--json", help="write the labels as JSON here")] = None,
    export: Annotated[Path | None, typer.Option(help="write Neuronpedia-format explanation rows here")] = None,
) -> None:
    """Label features from their top activating examples and score each label on held-out examples."""
    from circuitkit.labels import FeatureData, calibration, label_features, make_llm, score_labels
    from circuitkit.labels.neuronpedia import compare, export_explanations, feature_data
    from circuitkit.report import render_labels

    llm = make_llm(provider, model_name, cache)
    data: list[FeatureData] = []
    theirs: dict[str, str] = {}
    theirs_model: dict[str, str] = {}
    model_id = source = ""
    if neuronpedia:
        model_id, source = neuronpedia.split("/", 1)
        for idx in (features or "").split(","):
            fd, labs = feature_data(model_id, source, int(idx))
            data.append(fd)
            if labs:
                theirs[fd.unit], theirs_model[fd.unit] = labs[0]["description"], labs[0]["model"]
    elif records:
        data = [FeatureData.model_validate(x) for x in json.loads(records.read_text(encoding="utf-8"))]
    else:
        raise typer.BadParameter("give --neuronpedia with --features, or --records")
    ours = label_features(llm, data, score=score)
    comps = []
    if compare_neuronpedia and theirs:
        comps = compare(ours, score_labels(llm, data, theirs, score=score), theirs_model)
    for lab in ours:
        typer.echo(
            f"  {lab.unit:>10}  {lab.score if lab.score is not None else float('nan'):.2f}  {lab.label or lab.error}"
        )
    u = llm.usage
    typer.echo(
        f"API calls: {u.calls} (cache hits: {u.cached_calls}), {u.input_tokens}+{u.output_tokens} tokens, ${u.usd:.4f}"
    )
    if export and neuronpedia:
        export_explanations(ours, model_id, source, export)
    _write(
        json.dumps([x.model_dump() for x in ours], indent=1),
        render_labels(ours, calibration(ours), comps),
        json_out,
        html,
    )


if __name__ == "__main__":  # pragma: no cover
    app()
