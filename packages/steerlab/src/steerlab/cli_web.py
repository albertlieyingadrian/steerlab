"""`steerlab serve`: the steering demo as a web app (needs the `web` extra).

    steerlab serve                                   Qwen3-8B, the English-only prompt, http://127.0.0.1:8765
    steerlab serve --model qwen --port 9000          another downloaded model and port
    steerlab serve --host 0.0.0.0                    let other devices on the network in (prints a warning)

Visitors chat at `/`; the presenter steers at `/admin`. Only models already in the Hugging Face cache can be loaded or
swapped to, and nothing is downloaded. FastAPI and uvicorn are imported inside the command, so `steerlab --help`
stays light.
"""

from __future__ import annotations

import contextlib
import errno
import secrets
import socket
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from steerlab import directions as direction_lib
from steerlab import loading

if TYPE_CHECKING:
    from steerlab.control import ControlState
    from steerlab.loading import CatalogEntry

app = typer.Typer()


def lan_hosts(host: str) -> list[str]:
    """The names a LAN visitor's browser may send as `Host`: the bound address, this machine's IPv4 addresses, and
    `<hostname>.local`. Never `*`."""
    names = {host} if host not in ("0.0.0.0", "::") else set()
    hostname = socket.gethostname()
    names.add(hostname if hostname.endswith(".local") else f"{hostname}.local")
    with contextlib.suppress(OSError):
        names.update(str(info[4][0]) for info in socket.getaddrinfo(hostname, None, socket.AF_INET))
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:  # the address of the interface that routes outward (connecting a UDP socket sends no packet)
        probe.connect(("192.0.2.1", 9))
        names.add(probe.getsockname()[0])
    except OSError:
        pass
    finally:
        probe.close()
    return sorted(names)


def port_free(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError as error:
            if error.errno == errno.EADDRINUSE:
                return False
            raise
    return True


def load_entry(entry: CatalogEntry, on_message: Callable[[str], None]) -> tuple[Any, ControlState]:
    """Load a catalogued model from the cache only: its SAE layer when cached, and every saved direction."""
    sae_layers = [entry.sae.layer] if entry.sae is not None else []
    release = entry.sae.release if entry.sae is not None else None
    return loading.load_session(
        entry.key, sae_layers, release, entry.direction_specs, on_message=on_message, local_only=True
    )


def feature_label(model_id: str, release: str, layer: int, feature: int) -> str | None:
    from steerlab.features.cache import FeatureCache

    return FeatureCache().label(model_id, release, layer, feature)


@app.command()
def serve(
    model: Annotated[str, typer.Option("--model", "-m", help="A downloaded model key (qwen8b, qwen, gemma, ...).")] = (
        "qwen8b"
    ),
    system_file: Annotated[
        Path | None,
        typer.Option(
            "--system-file",
            help="One system prompt for every model. Default: each model's tuned English-only prompt (steerlab.demo).",
        ),
    ] = None,
    directions_dir: Annotated[
        Path, typer.Option("--directions-dir", help="Saved directions, one folder per model key.")
    ] = direction_lib.DEFAULT_OUTPUT_DIR,
    host: Annotated[str, typer.Option("--host", help="Address to bind. Not loopback = reachable on the network.")] = (
        "127.0.0.1"
    ),
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8765,
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens", min=1, help="Longest reply, in tokens.")] = 300,
    admin_token: Annotated[
        str | None,
        typer.Option(
            "--admin-token",
            help="Token /admin requires. Default: a random one when --host reaches beyond this machine, none on "
            "loopback.",
        ),
    ] = None,
) -> None:
    """Run the steering demo: a visitor chat at / and the admin desk at /admin."""
    try:
        import uvicorn

        from steerlab.web.app import LOOPBACK_HOSTS, create_app
        from steerlab.web.broker import Broker
        from steerlab.web.worker import Worker
    except ImportError as error:
        typer.echo(f"steerlab: `serve` needs the web extra ({error.name} is missing): pip install 'steerlab[web]'")
        raise typer.Exit(code=2) from None

    from steerlab import demo

    prompt_for: Any = demo.prompt_for
    system_prompt = demo.prompt_for(model)
    if system_file is not None:
        try:
            system_prompt = system_file.read_text(encoding="utf-8").rstrip()
        except OSError as error:
            typer.echo(f"steerlab: cannot read the system prompt {system_file}: {error.strerror or error}")
            raise typer.Exit(code=2) from None
        prompt_for = None
    keys = [entry.key for entry in loading.catalog(directions_dir)]
    if model not in keys:
        downloaded = ", ".join(keys) or "none"
        typer.echo(f"steerlab: {model!r} is not downloaded (downloaded: {downloaded}); serve never downloads a model")
        raise typer.Exit(code=2)
    if not port_free(host, port):
        typer.echo(f"steerlab: port {port} is already in use; pass --port")
        raise typer.Exit(code=2)

    hosts = list(LOOPBACK_HOSTS)
    admin_host = "localhost"
    if host.strip("[]") not in {name.strip("[]") for name in LOOPBACK_HOSTS}:
        reachable = lan_hosts(host)
        hosts += reachable
        admin_token = admin_token or secrets.token_urlsafe(16)
        admin_host = next((name for name in reachable if name[0].isdigit()), reachable[0] if reachable else host)
        typer.echo(
            f"steerlab: listening on {host}, reachable as {', '.join(reachable)}. /admin needs the token in the link "
            "below; share the visitor link, never the admin one."
        )
    worker = Worker(
        catalog=lambda: loading.catalog(directions_dir),
        load=load_entry,
        unload=loading.unload,
        system_prompt=system_prompt,
        broker=Broker(),
        initial_model=model,
        max_new_tokens=max_new_tokens,
        feature_label=feature_label,
        presets=demo.presets(),
        prompt_for=prompt_for,
    )
    application = create_app(worker, hosts=hosts, port=port, admin_token=admin_token)
    visitor_host = "127.0.0.1" if admin_host == "localhost" else admin_host
    admin_link = f"http://{admin_host}:{port}/admin" + (f"?token={admin_token}" if admin_token else "")
    typer.echo(f"steerlab: loading {model}; visitors: http://{visitor_host}:{port}/  admin: {admin_link}")
    uvicorn.run(application, host=host, port=port, log_level="warning", timeout_graceful_shutdown=2)
