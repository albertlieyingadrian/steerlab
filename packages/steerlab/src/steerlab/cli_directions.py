"""`steerlab directions …`: build and list the difference-of-means directions.

    steerlab directions build --model qwen                 all three concepts at the default layer
    steerlab directions build --model gemma -c chinese     one concept
    steerlab directions build --model tiny --out /tmp/d    offline smoke run on sandbagkit's tiny model
    steerlab directions list --model qwen                  present / missing, with the quality numbers

Mounted by `steerlab.cli` as the `directions` sub-command. Heavy imports (torch, transformers, sandbagkit.probing)
happen inside `build`; `list` reads only the JSON files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from steerlab import defaults
from steerlab.data import CONCEPTS

app = typer.Typer(
    name="directions",
    help="Difference-of-means directions (sandbagging, refusal, answer in Chinese) per model.",
    no_args_is_help=True,
    add_completion=False,
)

DEFAULT_OUT = str(defaults.directions_dir())

ModelOption = Annotated[
    str,
    typer.Option(
        "--model", "-m", help="qwen, gemma, an extra model (qwen8b, gemma2-2b, gemma2-9b, llama8b) or tiny (smoke)."
    ),
]
OutOption = Annotated[
    Path, typer.Option("--out", help="Root folder; directions go to <out>/<model>/<concept>.npz + .json.")
]


@app.command()
def build(
    model: ModelOption = "qwen",
    concept: Annotated[
        list[str] | None,
        typer.Option("--concept", "-c", help="Concept to build (repeatable). Default: all three."),
    ] = None,
    layer: Annotated[int | None, typer.Option("--layer", help="Decoder layer. Default: the model's SAE layer.")] = None,
    out: OutOption = Path(DEFAULT_OUT),
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 8,
    max_length: Annotated[int, typer.Option("--max-length", min=8)] = 512,
) -> None:
    """Build the directions from the shipped contrast pairs and save them with their quality numbers."""
    from steerlab import directions

    try:
        directions.check_model_key(model)
        names = directions.check_concepts(concept or list(CONCEPTS))
    except KeyError as error:
        typer.echo(f"error: {error.args[0]}", err=True)
        raise typer.Exit(code=2) from None

    typer.echo(f"model: {model}  out: {out}")
    built = directions.build_directions(
        model, names, out_dir=out, layer=layer, batch_size=batch_size, max_length=max_length
    )
    for item in built:
        meta = item.vector.metadata
        typer.echo(
            f"{item.concept:<12} layer {meta['layer']}, {meta['n_pairs']} pairs, "
            f"{directions.format_quality(meta)}  ->  {item.path}.npz"
        )


@app.command("list")
def list_(model: ModelOption = "qwen", out: OutOption = Path(DEFAULT_OUT)) -> None:
    """Report which directions exist for a model, with their quality numbers; name the command for missing ones."""
    from steerlab import directions

    try:
        directions.check_model_key(model)
    except KeyError as error:
        typer.echo(f"error: {error.args[0]}", err=True)
        raise typer.Exit(code=2) from None

    for status in directions.direction_status(model, out_dir=out):
        if status.present:
            typer.echo(
                f"{status.concept:<12} present  layer {status.metadata.get('layer')}, "
                f"{directions.format_quality(status.metadata)}  ({status.path}.npz)"
            )
        else:
            typer.echo(
                f"{status.concept:<12} missing  build with: {directions.build_command(model, status.concept, out)}"
            )
