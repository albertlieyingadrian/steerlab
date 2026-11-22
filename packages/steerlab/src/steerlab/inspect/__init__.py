"""steerlab.inspect: type words, watch a model arrive at its next word layer by layer (`steerlab inspect`).

    steerlab inspect                         the live explorer on Qwen2.5-0.5B-Instruct, http://127.0.0.1:8770
    steerlab inspect --export out/           the preset prompts as a static explorer (no server needed)

    from steerlab.inspect import Inspector, Edit
    inspector = Inspector.load()                            # Qwen2.5-0.5B-Instruct on the CPU, from the cache
    inspector.run("The capital of France is")["layers"]     # one row per layer: attention, neurons, logit lens

The views (`core`): the run (guesses, the target, each layer's share of the target's logit and its logit-lens guess),
a layer (its heads and top neurons), a neuron (the words it writes, how it fires on each token), find (neurons that
write a word), edit (switch neurons off, flip or scale them and compare) and dose (scale one neuron ×0 … ×8 against
20 random neurons of the same layer). `app` serves them as JSON beside one page; `export` writes the presets as JSON
the same page reads with no server. Inspired by neuronaut (github.com/andrew-morris/neuronaut).

Rooms. The page is a set of rooms in a top-level nav (Trace, Lens, Units, Edit, Workbench, Model), each at its own
URL (`/lens`, `/units/qwen3-8b/L8/n344+`; `/` is Trace). Keys: `g` then the room's key; `?` lists them.

How to add a room (nothing else to register; the shell finds it at startup):

1. `rooms/<id>.py`:  `ROOM = {"id": "<id>", "title": "<Title>", "order": <int>, "key": "<one letter>"}` and
   `def register(app, ctx)`, which adds routes under `/api/<id>/...`. `ctx` is a `RoomContext`: `ctx.inspector` (the
   loaded model), `ctx.view(fn)` (runs `fn` under the model lock, a `ValueError` becomes a 400), `ctx.lock`,
   `ctx.model_id`, `ctx.room_cache("<id>")` (a cache directory per room and model) and `ctx.rooms` (the nav).
   Taken: trace 10 t, lens 20 l, units 30 u, edit 40 e, workbench 50 w, model 60 m.
2. `static/rooms/<id>.js`: an ES module with `mount(el, api, route)` and `unmount()`, and optionally `update(route)`
   for a new deep link inside the room. `route = {room, path, params, hash, url}`; `api.get("<id>/x")`,
   `api.post("<id>/x", body)`, `api.navigate(url)`, `api.href(room, ...parts)`, `api.el` (DOM builder that inserts
   text with textContent), `api.show(message)`. Plain `<a href="/units/...">` links navigate in place.
3. `static/rooms/<id>.css` (optional): reuse the tokens and classes in `inspect.css` (`.panel`, `.facts`, `.badge`,
   `.tok`, tables, `.pos`/`.neg`); orange is the effect, blue the control, grey the random control.
4. Heavy numerics go in a plain module outside `inspect` (torch imported inside functions), tested on the tiny model
   from `tests/inspector/conftest.py`; add `tests/inspector/test_<id>.py` for the routes.

Importing this package imports neither torch nor FastAPI; `core` imports torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from steerlab.inspect.core import Edit, Inspector

__all__ = ["Edit", "Inspector"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from steerlab.inspect import core

        return getattr(core, name)
    raise AttributeError(name)
