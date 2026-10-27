"""sandbagkit: detectors for AI sandbagging and evaluation awareness.

This file stays tiny on purpose: `import sandbagkit` must be instant and must never pull in torch.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    # The version is written once, in pyproject.toml. This reads it back from the installed metadata.
    __version__ = version("sandbagkit")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
