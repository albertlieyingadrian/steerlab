"""The `steerlab` command line.

    steerlab --version         print the installed version
    steerlab doctor            device, dtype, what is cached (models, SAE layers, judge), Gemma access, directions
    steerlab doctor --online   also ask the Hub whether this account may download the gated Gemma model
    steerlab serve             the steering demo as a web app: visitor chat at /, admin desk at /admin

Heavy libraries are imported inside the commands that need them, so `steerlab --help` stays instant. `doctor` reads
the Hugging Face cache offline and never prints a token.
"""

from __future__ import annotations

import importlib
import importlib.util
import platform
from fnmatch import fnmatch
from types import ModuleType
from typing import Annotated

import typer

from steerlab import __version__, cli_directions, cli_engine, defaults, directions, hub
from steerlab.saes import gemma_scope, qwen_scope

app = typer.Typer(
    name="steerlab",
    help="Live activation steering on small open models.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(cli_engine.app)
app.add_typer(cli_engine.schema_app, name="schema", help="Print the published schemas (`schema control`).")
app.add_typer(
    cli_engine.control_app, name="control", help="Check or change a control file (`control validate|set|off`)."
)
app.add_typer(cli_directions.app, name="directions", help="Build and list difference-of-means directions.")


def optional_module(name: str) -> ModuleType | None:
    """Import a sub-command's module, or None when that module is not part of this install.

    Only the module's own absence is tolerated: an error inside a module that is present, or a missing third-party
    package it needs, still raises, so a broken sub-command never disappears silently.
    """
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name is not None and (name == error.name or name.startswith(error.name + ".")):
            return None
        raise


# Sub-commands whose modules may be absent from a smaller install: (module, attribute, command name, help).
OPTIONAL_COMMANDS: list[tuple[str, str, str | None, str]] = [
    ("steerlab.cli_web", "app", None, ""),
]

EXTRAS: dict[str, list[str]] = {
    "models": ["torch", "transformers", "safetensors", "huggingface_hub", "sandbagkit"],
    "web": ["fastapi", "uvicorn"],
}

# More sub-commands and extras, when this install has `steerlab.extra_commands` (COMMANDS and EXTRAS, same shapes).
_extra = optional_module("steerlab.extra_commands")
if _extra is not None:
    OPTIONAL_COMMANDS += _extra.COMMANDS
    EXTRAS.update(_extra.EXTRAS)

for _module_name, _attribute, _command, _help in OPTIONAL_COMMANDS:
    _module = optional_module(_module_name)
    if _module is None:
        continue
    if _command is None:
        app.add_typer(getattr(_module, _attribute))
    else:
        app.add_typer(getattr(_module, _attribute), name=_command, help=_help)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"steerlab {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version_callback, is_eager=True, help="Show the version and exit.")
    ] = False,
) -> None:
    """Live activation steering on small open models."""


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _device_and_dtype() -> tuple[str, str]:
    """The device the engine would run on and the dtype it would load models in (float32 on CPU, bfloat16 elsewhere)."""
    if not _installed("torch"):
        hint = 'n/a (install "steerlab[models]")'
        return hint, hint
    import torch

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    dtype = "float32" if device == "cpu" else "bfloat16"
    return f"{device} (torch {torch.__version__})", dtype


def _status(repo_id: str) -> str:
    """'cached  (3.0 MB)' or 'not cached  (~4.0 GB to download)'."""
    size = hub.cached_bytes(repo_id)
    if size is not None:
        return f"cached  ({hub.format_size(size)})"
    approx = defaults.APPROX_DOWNLOAD_BYTES.get(repo_id)
    return f"not cached  (~{hub.format_size(approx)} to download)" if approx else "not cached"


def _sae_lines(release: str, default_file: str, pattern: str) -> list[str]:
    """One line for the default layer file, then one per other cached layer file of the release."""
    files = hub.cached_files(release)
    per_layer = defaults.APPROX_DOWNLOAD_BYTES.get(release)
    approx = f"not cached  (~{hub.format_size(per_layer)} per layer)" if per_layer else "not cached"
    status = f"cached  ({hub.format_size(files[default_file])})" if default_file in files else approx
    lines = [f"  {release}  {default_file}  {status}"]
    others = sorted(
        (name for name in files if name != default_file and fnmatch(name, pattern)),
        key=lambda name: hub.layer_in_filename(name) or 0,
    )
    lines.extend(f"  {release}  {name}  cached  ({hub.format_size(files[name])})" for name in others)
    return lines


@app.command()
def doctor(
    online: Annotated[
        bool,
        typer.Option(
            "--online",
            help="Also ask the Hub (one request per gated repo) whether this account may download it. "
            "Default: offline, nothing is contacted.",
        ),
    ] = False,
) -> None:
    """Report the device, what is cached, what each download would cost, Gemma access, and the directions present."""
    typer.echo(f"steerlab {__version__}")
    typer.echo(f"python {platform.python_version()} on {platform.platform()}")
    typer.echo("")
    typer.echo("extras:")
    for extra, modules in EXTRAS.items():
        missing = [m for m in modules if not _installed(m)]
        hint = f'missing {", ".join(missing)}  ->  pip install "steerlab[{extra}]"'
        typer.echo(f"  {extra:<8} {hint if missing else 'installed'}")
    typer.echo("")
    device, dtype = _device_and_dtype()
    typer.echo(f"device: {device}")
    typer.echo(f"dtype:  {dtype}")
    typer.echo(f"hf token: {'present' if hub.token_present() else 'absent'} (never printed)")
    typer.echo(f"hub cache: {hub.hub_cache_dir()}")
    typer.echo("")
    typer.echo("models:")
    for repo_id in defaults.MODELS.values():
        line = f"  {repo_id:<42} {_status(repo_id)}"
        if repo_id in defaults.GATED_REPOS:
            if online:
                access = "yes" if hub.has_access(repo_id) else f"no (accept the licence at {hub.licence_url(repo_id)})"
            else:
                access = "not checked (pass --online)"
            line += f"  gated: access {access}"
        typer.echo(line)
    typer.echo("judge (only for `sweep judge` and on-demand Qwen labels):")
    typer.echo(f"  {defaults.JUDGE_MODEL:<42} {_status(defaults.JUDGE_MODEL)}")
    typer.echo("sae layers:")
    for line in _sae_lines(
        defaults.QWEN_SCOPE_RELEASE, qwen_scope.filename(defaults.QWEN_SCOPE_LAYER), "layer*.sae.pt"
    ):
        typer.echo(line)
    for line in _sae_lines(
        defaults.GEMMA_SCOPE_RELEASE, gemma_scope.params_path(defaults.GEMMA_SCOPE_LAYER), "*/params.safetensors"
    ):
        typer.echo(line)
    # Where `directions build` writes (relative to the directory this runs from), per model key and concept.
    typer.echo(f"directions ({directions.DEFAULT_OUTPUT_DIR.resolve()}):")
    for model_key in directions.MODEL_KEYS:
        for status in directions.direction_status(model_key, out_dir=directions.DEFAULT_OUTPUT_DIR):
            typer.echo(f"  {model_key:<6} {status.concept:<12} {'present' if status.present else 'missing'}")
