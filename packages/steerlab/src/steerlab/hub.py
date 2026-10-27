"""Hugging Face Hub helpers that never print a token.

Two jobs:

1. Read the Hub cache OFFLINE (`~/.cache/huggingface/hub`, or `HF_HUB_CACHE` / `HF_HOME`) so `steerlab doctor` can say
   what is already on disk without a network request.
2. Guard downloads: `ensure_access` asks the Hub whether this account may read a repo and, for a gated repo it may not,
   raises an error that names the licence page. The token itself is never read into steerlab, only its presence.

huggingface_hub is imported lazily inside the functions that talk to the Hub, so this module stays light.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

HF_HUB_URL = "https://huggingface.co"


class GatedRepoAccessError(PermissionError):
    """A gated (or missing) repo rejected this account before any download started."""


def licence_url(repo_id: str) -> str:
    """The Hub page where the licence is accepted (the model card)."""
    return f"{HF_HUB_URL}/{repo_id}"


def hub_cache_dir() -> Path:
    """Where huggingface_hub stores models, resolved the way huggingface_hub.constants does, without importing it."""
    if env := os.environ.get("HF_HUB_CACHE"):
        return Path(env).expanduser()
    if home := os.environ.get("HF_HOME"):
        return Path(home).expanduser() / "hub"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path("~/.cache").expanduser()
    return base / "huggingface" / "hub"


def steerlab_cache_dir() -> Path:
    """Where steerlab keeps labels, calibration scales and directions (`STEERLAB_CACHE`, default ~/.cache/steerlab)."""
    if env := os.environ.get("STEERLAB_CACHE"):
        return Path(env).expanduser()
    return Path("~/.cache/steerlab").expanduser()


def _token() -> str | None:
    """The token to pass explicitly to Hub calls, or None to let huggingface_hub resolve it itself.

    huggingface_hub reads `HF_TOKEN` (and the token file) on its own. This repo's `.env` names the token
    `HUGGINGFACE_TOKEN`, which huggingface_hub ignores, so that one is forwarded as `token=` when `HF_TOKEN` is unset.
    The value is only ever handed to huggingface_hub: never printed, logged or written.
    """
    if os.environ.get("HF_TOKEN"):
        return None
    return os.environ.get("HUGGINGFACE_TOKEN") or None


def token_present() -> bool:
    """Whether a Hub token is configured, by existence only. Its value is never printed."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"):
        return True
    if env := os.environ.get("HF_TOKEN_PATH"):
        return Path(env).expanduser().is_file()
    home = os.environ.get("HF_HOME")
    base = Path(home).expanduser() if home else Path("~/.cache/huggingface").expanduser()
    return (base / "token").is_file()


def repo_cache_dir(repo_id: str, cache_dir: Path | None = None) -> Path:
    """The cache folder of one model repo: `<cache>/models--<org>--<name>`."""
    return (cache_dir or hub_cache_dir()) / ("models--" + repo_id.replace("/", "--"))


def cached_files(repo_id: str, cache_dir: Path | None = None) -> dict[str, int]:
    """Files of a cached repo mapped to their byte sizes (symlinks into blobs/ resolved). Empty when not cached.

    Every snapshot is scanned; when a file exists in several revisions the most recently modified snapshot wins.
    """
    snapshots = repo_cache_dir(repo_id, cache_dir) / "snapshots"
    if not snapshots.is_dir():
        return {}
    files: dict[str, int] = {}
    for snapshot in sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime):
        if not snapshot.is_dir():
            continue
        for path in snapshot.rglob("*"):
            if path.is_file():  # follows the symlink into blobs/, so incomplete downloads (no blob) are skipped
                files[path.relative_to(snapshot).as_posix()] = path.stat().st_size
    return files


def cached_file(repo_id: str, filename: str, cache_dir: Path | None = None) -> Path | None:
    """The on-disk path of one cached file, or None when it is not in the cache."""
    snapshots = repo_cache_dir(repo_id, cache_dir) / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = [p for p in snapshots.glob(f"*/{filename}") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def cached_bytes(repo_id: str, cache_dir: Path | None = None) -> int | None:
    """Total bytes of a cached repo, or None when nothing of it is cached."""
    files = cached_files(repo_id, cache_dir)
    return sum(files.values()) if files else None


def format_size(nbytes: float) -> str:
    """Human sizes in binary units: '16.4 GB', '537 MB', '3.0 MB'."""
    gb = nbytes / 1024**3
    if gb >= 1:
        return f"{gb:.1f} GB"
    mb = nbytes / 1024**2
    return f"{mb:.1f} MB" if mb < 10 else f"{mb:.0f} MB"


def ensure_access(repo_id: str) -> None:
    """Ask the Hub (one request) whether this account may read `repo_id`; raise GatedRepoAccessError if not.

    The error names the licence page and the login command, never the token.
    """
    import huggingface_hub
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    try:
        huggingface_hub.auth_check(repo_id, token=_token())
    except GatedRepoError:
        raise GatedRepoAccessError(
            f"{repo_id} is gated and this Hugging Face account has no access to it (or no token is configured). "
            f"Accept the licence at {licence_url(repo_id)}, then run `hf auth login`, then retry. "
            "steerlab never downloads a gated model on your behalf and never prints your token."
        ) from None
    except RepositoryNotFoundError:
        raise GatedRepoAccessError(
            f"{repo_id} was not found on the Hub, or it is private and this account cannot see it: "
            f"{licence_url(repo_id)}"
        ) from None


def has_access(repo_id: str) -> bool:
    """yes/no form of `ensure_access`, for `doctor --online`."""
    try:
        ensure_access(repo_id)
    except GatedRepoAccessError:
        return False
    return True


def download(repo_id: str, filename: str, *, revision: str | None = None) -> Path:
    """Return the local path of `filename` from `repo_id`: the cached copy if present, else download after the guard.

    The cache is checked first so loading an SAE that is already on disk needs no network and no token.
    """
    cached = cached_file(repo_id, filename)
    if cached is not None:
        return cached
    ensure_access(repo_id)
    import huggingface_hub

    return Path(huggingface_hub.hf_hub_download(repo_id, filename, revision=revision, token=_token()))


def layer_in_filename(filename: str) -> int | None:
    """The layer number in an SAE file path (`layer14.sae.pt`, `resid_post/layer_13_width_16k_l0_medium/...`)."""
    match = re.search(r"layer_?(\d+)", filename)
    return int(match.group(1)) if match else None
