"""Raw loaders for the two SAE releases: Qwen-Scope (TopK, `.pt`) and Gemma Scope 2 (JumpReLU, safetensors).

Neither this package nor its submodules import torch at import time: the loaders import it inside `load_*`, so the
pure path helpers (`qwen_scope.filename`, `gemma_scope.params_path`) work on the torch-free base install, which is
what `steerlab doctor` relies on.
"""

from __future__ import annotations

__all__ = ["common", "gemma_scope", "qwen_scope"]
