"""steerlab.workbench: chat with a model under steering, on a branching loom, with live concept probes.

- `loom`: the conversation as a tree of turns (reroll, fork, compare); pure Python.
- `artifacts`: directions, probes, lenses and edits saved as JSON + npz; export and import as a zip.
- `probes`: the bundled concept probes (difference of means) and their random-direction baseline.
- `edits`: apply an Edit-room artifact (H = I - k d d^T on chosen units' write columns) for one reply, then restore.
- `session`: the Workbench itself: generate a reply with steering, edits and per-token readings.
- `openai_api`: the OpenAI-compatible request and response shapes for /v1/chat/completions.

Importing this package imports no torch; `session` and the fitting functions import it when called.
"""
