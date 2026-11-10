"""`steerlab inspect`: type words, watch the model pick its next word layer by layer (`models` and `web` extras).

    steerlab inspect                              Qwen2.5-0.5B-Instruct on the CPU, http://127.0.0.1:8770
    steerlab inspect --model Qwen/Qwen2.5-1.5B-Instruct --port 9000
    steerlab inspect --export site/explorer       the preset prompts as a static page (no server), then exit

Only models already in the Hugging Face cache are loaded unless `--download` is passed. Heavy imports happen inside
the command, so `steerlab --help` stays light.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from steerlab.cli_web import port_free

app = typer.Typer()

DEFAULT_PORT = 8770


@app.command()
def inspect(
    model: Annotated[str, typer.Option("--model", "-m", help="A Hugging Face model id (Qwen2/Qwen3/Llama-style).")] = (
        "Qwen/Qwen2.5-0.5B-Instruct"
    ),
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = DEFAULT_PORT,
    export: Annotated[
        Path | None, typer.Option("--export", help="Write the preset prompts as a static explorer here, then exit.")
    ] = None,
    download: Annotated[bool, typer.Option("--download", help="Allow downloading the model if it is not cached.")] = (
        False
    ),
) -> None:
    """Open the layer explorer: tokens, the layer stack, heads and neurons, find, edit and dose tests."""
    try:
        import uvicorn

        from steerlab.inspect.app import LOOPBACK_HOSTS, create_app
        from steerlab.inspect.core import Inspector
        from steerlab.inspect.export import export as write_export
    except ImportError as error:
        typer.echo(
            f"steerlab: `inspect` needs the models and web extras ({error.name} is missing): "
            "pip install 'steerlab[models,web]'"
        )
        raise typer.Exit(code=2) from None

    if export is None and not port_free("127.0.0.1", port):
        typer.echo(f"steerlab: port {port} is already in use; pass --port")
        raise typer.Exit(code=2)
    typer.echo(f"steerlab: loading {model} on the CPU")
    try:
        inspector = Inspector.load(model, local_only=not download)
    except OSError as error:
        typer.echo(f"steerlab: cannot load {model} ({error}); pass --download to fetch it")
        raise typer.Exit(code=2) from None
    if export is not None:
        write_export(inspector, export, on_message=typer.echo)
        return
    typer.echo(f"steerlab: layer explorer at http://127.0.0.1:{port}/")
    uvicorn.run(create_app(inspector, hosts=LOOPBACK_HOSTS), host="127.0.0.1", port=port, log_level="warning")
