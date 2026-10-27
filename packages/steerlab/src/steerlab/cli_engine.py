"""The engine's commands, mounted into the main `steerlab` app by `cli.py` (`app.add_typer(cli_engine.app)`).

    steerlab chat --model qwen --control control.json [--sae-layer 14] [--direction chinese=PATH]
    steerlab generate "PROMPT" --model qwen --control control.json [--events events.json] [--json]
    steerlab schema control                 print the control file's JSON Schema
    steerlab control validate control.json  check a control file (schema, and layer indexes for a known model)
    steerlab control set chinese=1.5        move a fader (temp file + rename), printing before and after
    steerlab control set feature:24946=4    the same for an SAE feature of the model's own release
    steerlab control off                    clear every entry (steering off)

`--control` names the file the engine re-reads before every token. `--events` scripts control changes
at fixed token indices for the recorded demo. `--json` prints the text and the per-token timeline for
agents.
Heavy libraries are imported inside the commands, so `steerlab --help` and the two schema commands stay light.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from steerlab import defaults, loading
from steerlab.control import (
    MODEL_LAYERS,
    ControlEntry,
    ControlFile,
    ControlValidationError,
    ControlWatcher,
    DirectionSource,
    SAESource,
    control_json_schema,
    parse_control,
    validate_layers,
)
from steerlab.loading import DEFAULT_SAE_LAYERS, default_release, resolve_model

if TYPE_CHECKING:
    from steerlab.control import ControlState
    from steerlab.decode import TimelineEntry

app = typer.Typer(help="Steer a model live: chat, generate, and the control file schema.")
schema_app = typer.Typer(help="Print the published schemas.", no_args_is_help=True)
control_app = typer.Typer(help="Check control files.", no_args_is_help=True)
app.add_typer(schema_app, name="schema")
app.add_typer(control_app, name="control")

ModelOption = Annotated[
    str, typer.Option("--model", "-m", help="qwen, gemma, a Hub model id, or a local model directory.")
]
ControlOption = Annotated[
    Path | None, typer.Option("--control", "-c", help="Control file to re-read before every generated token.")
]
SystemOption = Annotated[
    str | None, typer.Option("--system", help="System prompt (folded into the first user turn on Gemma).")
]
SaeLayerOption = Annotated[
    list[int], typer.Option("--sae-layer", help="Load this layer of the model's SAE release (repeatable).")
]
SaeReleaseOption = Annotated[
    str | None, typer.Option("--sae-release", help="SAE release for --sae-layer. Default: the model's own release.")
]
DirectionOption = Annotated[
    list[str],
    typer.Option(
        "--direction", "-d", help="NAME=PATH: load a saved sandbagkit steering vector as a direction (repeatable)."
    ),
]
MaxNewTokensOption = Annotated[int, typer.Option("--max-new-tokens", min=1)]


def say(message: str) -> None:
    """Engine messages go to stderr, so `generate --json` leaves stdout as pure JSON."""
    typer.echo(f"steerlab: {message}", err=True)


def load_session(
    model: str,
    sae_layers: list[int],
    sae_release: str | None,
    directions: list[str],
    *,
    on_message: Callable[[str], None] = say,
) -> tuple[Any, ControlState]:
    """`loading.load_session` for the CLI: a load problem is printed and exits 2 before any generation starts."""
    try:
        return loading.load_session(model, sae_layers, sae_release, directions, on_message=on_message, on_progress=say)
    except loading.LoadError as error:
        typer.echo(f"steerlab: {error}", err=True)
        raise typer.Exit(code=2) from None


def require_control_file(control: Path | None) -> None:
    """A `--control` path that does not exist is almost always a typo: say so now, not after the model loads."""
    if control is not None and not control.is_file():
        typer.echo(f"steerlab: control file {control} does not exist (--control names the file to re-read)", err=True)
        raise typer.Exit(code=2)


def print_change(entry: TimelineEntry) -> None:
    typer.echo(f"\n{entry.describe()}\n  (earlier steered text stays in the context)")


@app.command()
def generate(
    prompt: Annotated[str, typer.Argument(help="The user message.")],
    model: ModelOption = "qwen",
    control: ControlOption = None,
    events: Annotated[
        Path | None, typer.Option("--events", help="Scripted control changes: a JSON list of {token_index, control}.")
    ] = None,
    system: SystemOption = None,
    max_new_tokens: MaxNewTokensOption = 200,
    sae_layer: SaeLayerOption | None = None,
    sae_release: SaeReleaseOption = None,
    direction: DirectionOption | None = None,
    json_out: Annotated[
        bool, typer.Option("--json", help="Print {model, prompt, text, token_ids, timeline, messages} as JSON.")
    ] = False,
) -> None:
    """Generate one steered reply, streaming it (or, with --json, returning text plus the per-token timeline).

    With --json, `messages` lists every engine message of the run (skipped entries, a model mismatch, a file that
    failed to parse), so an agent can tell unsteered text from steered text without reading stderr.
    """
    from steerlab.decode import encode_prompt, greedy_generate, load_events, render_messages
    from steerlab.engine import SteeringSession

    require_control_file(control)
    event_list = None
    if events is not None:
        try:
            event_list = load_events(events)
        except (OSError, json.JSONDecodeError, ControlValidationError) as error:
            typer.echo(f"steerlab: {error}", err=True)
            raise typer.Exit(code=1) from None
    engine_messages: list[str] = []

    def record(message: str) -> None:
        engine_messages.append(message)
        say(message)

    backend, state = load_session(model, sae_layer or [], sae_release, direction or [], on_message=record)
    watcher = ControlWatcher(control, state) if control is not None else None
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    with SteeringSession(backend.model, backend.tokenizer, state) as session:
        text = render_messages(backend.tokenizer, messages, model_type=session.model_type)
        ids = encode_prompt(backend.tokenizer, text, session.device)
        if json_out:
            generation = greedy_generate(
                session, ids, max_new_tokens=max_new_tokens, watcher=watcher, events=event_list
            )
            payload = {"model": state.model, "prompt": prompt, **generation.to_dict(), "messages": engine_messages}
            typer.echo(json.dumps(payload))
            return
        greedy_generate(
            session,
            ids,
            max_new_tokens=max_new_tokens,
            watcher=watcher,
            events=event_list,
            on_token=lambda _index, piece: typer.echo(piece, nl=False),
            on_change=print_change,
        )
        typer.echo("")


@app.command()
def chat(
    model: ModelOption = "qwen",
    control: ControlOption = None,
    system: SystemOption = None,
    max_new_tokens: MaxNewTokensOption = 400,
    sae_layer: SaeLayerOption | None = None,
    sae_release: SaeReleaseOption = None,
    direction: DirectionOption | None = None,
) -> None:
    """Chat in the terminal while another terminal edits the control file. `r` restarts the reply, `c` clears the
    history, `q` quits."""
    from steerlab import decode
    from steerlab.engine import SteeringSession

    require_control_file(control)
    backend, state = load_session(model, sae_layer or [], sae_release, direction or [])
    watcher = ControlWatcher(control, state) if control is not None else None
    typer.echo(f"model: {state.model} ({state.n_layers} layers, {backend.device})")
    if control is not None:
        typer.echo(f'control file: {control} (re-read before every token; its "model" must be {state.model!r})')
    with SteeringSession(backend.model, backend.tokenizer, state) as session:
        decode.chat(
            session,
            max_new_tokens=max_new_tokens,
            system=system,
            watcher=watcher,
            write=lambda piece: typer.echo(piece, nl=False),
        )


@schema_app.command("control")
def schema_control() -> None:
    """Print the JSON Schema of the control file."""
    typer.echo(json.dumps(control_json_schema(), indent=2))


@control_app.command("validate")
def control_validate(
    path: Annotated[Path, typer.Argument(help="The control file to check.")],
    layers: Annotated[
        int | None,
        typer.Option(
            "--layers",
            help="Number of decoder layers to check indexes against. Default: known for the two supported models.",
        ),
    ] = None,
) -> None:
    """Check a control file against the schema and, for a known model, its layer count. Exit 1 on any problem."""
    try:
        parsed = parse_control(path.read_text())
    except OSError as error:
        typer.echo(f"{path.name}: {error.strerror or error}")
        raise typer.Exit(code=1) from None
    except ControlValidationError as error:
        typer.echo(f"{path.name}: {error}")
        raise typer.Exit(code=1) from None
    n_layers = layers if layers is not None else MODEL_LAYERS.get(parsed.model)
    problems = validate_layers(parsed, n_layers)
    for problem in problems:
        typer.echo(f"{path.name}: {problem}")
    if problems:
        raise typer.Exit(code=1)
    checked = f", layer indexes checked against {n_layers} layers" if n_layers is not None else ", layer count unknown"
    typer.echo(f"ok: {path.name} steers {parsed.model} with {len(parsed.entries)} entries{checked}")


ControlFileOption = Annotated[Path, typer.Option("--file", "-f", help="The control file to change.")]


def read_control_or_none(path: Path) -> ControlFile | None:
    """The file's current state, or None when it is missing or invalid (a writer starts over in both cases)."""
    try:
        return parse_control(path.read_text())
    except (OSError, ControlValidationError):
        return None


def show_control(label: str, path: Path) -> None:
    try:
        typer.echo(f"{label} {path.read_text().strip()}")
    except OSError:
        typer.echo(f"{label} (no {path.name} yet)")


def write_control(path: Path, control: ControlFile) -> None:
    """Write to a temp file and rename it over `path`, so a `chat` re-reading every token never sees half a file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(control.model_dump_json(exclude_none=True) + "\n")
    os.replace(tmp, path)


def control_model(model: str | None, current: ControlFile | None) -> str:
    """The model a control file command writes for: `--model`, else the file's, else Qwen3-1.7B."""
    if model:
        return resolve_model(model)
    return current.model if current else defaults.QWEN_MODEL


def setting_source(name: str, model_id: str, sae_layer: int | None) -> SAESource | DirectionSource:
    """`feature:24946` is that feature of the model's own SAE release; any other name is a direction."""
    kind, sep, feature = name.partition(":")
    if not (kind == "feature" and sep):
        return DirectionSource(kind="direction", name=name)
    release = default_release(model_id)
    layer = sae_layer if sae_layer is not None else DEFAULT_SAE_LAYERS.get(model_id)
    if release is None or layer is None:
        raise ValueError(f"{model_id} has no default SAE release, so feature:ID cannot name a feature on it")
    if not feature.isdigit():
        raise ValueError(f"{name!r}: the feature ID must be a whole number, e.g. feature:24946")
    return SAESource(kind="sae", release=release, layer=layer, feature=int(feature))


@control_app.command("set")
def control_set(
    settings: Annotated[
        list[str],
        typer.Argument(
            help="NAME=STRENGTH for a direction (chinese=1.5) or feature:ID=STRENGTH for an SAE feature "
            "(feature:24946=4); several allowed."
        ),
    ],
    path: ControlFileOption = Path("control.json"),
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="qwen, gemma or a Hub id. Default: the file's, else qwen.")
    ] = None,
    sae_layer: Annotated[
        int | None, typer.Option("--sae-layer", help="SAE layer for feature:ID. Default: 14 on Qwen, 13 on Gemma.")
    ] = None,
) -> None:
    """Move faders in a control file and print it before and after. Other entries are kept."""
    current = read_control_or_none(path)
    model_id = control_model(model, current)
    sources: list[tuple[SAESource | DirectionSource, float]] = []
    for setting in settings:
        name, sep, value = setting.partition("=")
        try:
            strength = float(value)
        except ValueError:
            sep = ""
        if not sep or not name:
            typer.echo(f"steerlab: {setting!r} is not NAME=STRENGTH (e.g. chinese=1.5 or feature:24946=4)")
            raise typer.Exit(code=1)
        try:
            sources.append((setting_source(name, model_id, sae_layer), strength))
        except ValueError as error:
            typer.echo(f"steerlab: {error}")
            raise typer.Exit(code=1) from None
    entries = list(current.entries) if current and current.model == model_id else []
    for source, strength in sources:
        for index, entry in enumerate(entries):
            if entry.source == source:
                entries[index] = entry.model_copy(update={"strength": strength})
                break
        else:
            entries.append(ControlEntry(source=source, strength=strength))
    show_control("before:", path)
    write_control(path, ControlFile(model=model_id, entries=entries))
    show_control("after: ", path)


@control_app.command("off")
def control_off(
    path: ControlFileOption = Path("control.json"),
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Model the file steers. Default: the file's, else qwen.")
    ] = None,
) -> None:
    """Remove every entry (steering off), keep the model (or set it), and print the file before and after."""
    current = read_control_or_none(path)
    model_id = control_model(model, current)
    show_control("before:", path)
    write_control(path, ControlFile(model=model_id))
    show_control("after: ", path)
