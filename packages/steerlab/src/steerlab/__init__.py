"""steerlab: live activation steering on small open models.

This file stays tiny on purpose: `import steerlab` must be instant and must never pull in torch.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("steerlab")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
