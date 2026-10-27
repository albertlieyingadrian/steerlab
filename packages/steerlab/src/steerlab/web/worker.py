"""The worker: one thread that owns the model and writes one reply at a time.

Think of a radio studio with one microphone. Visitors queue for it (the visitor queue, first come first served); the
producer's requests (change the prompt, redo, clear, swap the model) go to the front of the line (the admin queue);
and the sound desk (the steering faders, and the stop button) works while someone is on air, through the `Poller`
that `greedy_generate` checks before every token.

Only this thread touches the model, the `SteeringSession`, the `ControlState` and the conversations' turns: none of
them is thread-safe. The HTTP side calls the public methods below, which validate, then queue a command, push a
control change to the poller, or read a snapshot under the lock. Events go out two ways: every event to the
`Broker` (the admin mirror), and a visitor's own reply events to that request's `sink`.

    worker = Worker(catalog=..., load=..., unload=..., system_prompt=..., broker=Broker(), initial_model="qwen8b")
    worker.start()
    job = worker.submit("conversation-id", "Hello", sink=events.append)
    worker.set_strength("chinese", 1.5)    # applies from the next generated token
"""

from __future__ import annotations

import contextlib
import gc
import math
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from steerlab.control import ControlEntry, ControlFile, DirectionSource, SAESource
from steerlab.web.broker import Broker, Event
from steerlab.web.conversations import FULL_MESSAGE, MAX_PROMPT_TOKENS, ConversationStore, Message, MessageRejected

if TYPE_CHECKING:
    from collections.abc import Mapping

    from steerlab.control import ControlState
    from steerlab.decode import TimelineEntry
    from steerlab.demo import Preset
    from steerlab.engine import SteeringSession
    from steerlab.loading import CatalogEntry

Sink = Callable[[Event], None]
Loader = Callable[["CatalogEntry", Callable[[str], None]], tuple[Any, "ControlState"]]
Unloader = Callable[[Any, "ControlState"], None]
FeatureLabeler = Callable[[str, str, int, int], "str | None"]

QUEUE_CAPACITY = 20
MAX_FEATURES = 8
UNAVAILABLE = "The assistant is unavailable right now. Try again in a moment."


class ReplyCancelled(Exception):
    """Raised inside `greedy_generate` (from `Poller.poll`) to end the reply now. `reason` becomes `end.reason`."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class QueueFull(MessageRejected):
    """The visitor queue is at capacity."""


class Poller:
    """The per-token seam into a running reply: pending control changes and the cancel flag.

    The HTTP side calls `push` and `cancel`; the worker thread calls `poll` (through `greedy_generate`'s `watcher`)
    before each token and at every command boundary. The newest pushed control wins; nothing is applied off-thread.
    """

    def __init__(self, on_applied: Callable[[], None]) -> None:
        self._lock = threading.Lock()
        self._control: ControlFile | None = None
        self._cancel: str | None = None
        self._active: str | None = None
        self.state: ControlState | None = None
        self.on_applied = on_applied

    def push(self, control: ControlFile) -> None:
        with self._lock:
            self._control = control

    def discard(self) -> None:
        with self._lock:
            self._control = None

    def cancel(self, reason: str, conversation: str | None = None) -> bool:
        """Stop the running reply (only `conversation`'s, when given). False when nothing matching is running."""
        with self._lock:
            if self._active is None or (conversation is not None and conversation != self._active):
                return False
            self._cancel = reason
            return True

    def begin(self, conversation: str) -> None:
        with self._lock:
            self._active, self._cancel = conversation, None

    def end(self) -> None:
        with self._lock:
            self._active, self._cancel = None, None

    @property
    def active(self) -> str | None:
        with self._lock:
            return self._active

    def poll(self) -> bool:
        with self._lock:
            control, self._control = self._control, None
            cancel = self._cancel
        changed = False
        if control is not None and self.state is not None:
            self.state.forget_messages()
            self.state.apply(control)
            changed = True
            self.on_applied()
        if cancel is not None:
            raise ReplyCancelled(cancel)
        return changed


@dataclass
class Job:
    """A visitor message waiting for, or getting, its reply."""

    conversation: str
    text: str
    sink: Sink
    cancelled: bool = False


@dataclass
class _Loaded:
    entry: CatalogEntry
    backend: Any
    state: ControlState
    session: SteeringSession
    n_features: int | None = None


@dataclass
class _Faders:
    """The admin's faders: one strength per direction, plus the SAE features the admin added (id -> strength)."""

    directions: dict[str, float] = field(default_factory=dict)
    features: dict[int, float] = field(default_factory=dict)
    labels: dict[int, str] = field(default_factory=dict)
    modes: dict[int, str] = field(default_factory=dict)  # feature -> "clamp" for a preset's clamped feature


class Worker:
    def __init__(
        self,
        *,
        catalog: Callable[[], list[CatalogEntry]],
        load: Loader,
        unload: Unloader,
        system_prompt: str,
        broker: Broker,
        initial_model: str,
        max_new_tokens: int = 300,
        feature_label: FeatureLabeler | None = None,
        presets: Mapping[str, Preset] | None = None,
        prompt_for: Callable[[str], str] | None = None,
    ) -> None:
        """`presets` (by model key) put each model's tuned strengths and features on the desk when it loads;
        `prompt_for` gives each model its own system prompt on load (None: `system_prompt` for every model)."""
        self._catalog = catalog
        self._models: list[CatalogEntry] = catalog()
        self._load = load
        self._unload = unload
        self._default_prompt = system_prompt
        self._prompt = system_prompt
        self.broker = broker
        self.max_new_tokens = max_new_tokens
        self._feature_label = feature_label
        self._presets = dict(presets or {})
        self._prompt_for = prompt_for
        self.store = ConversationStore()
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._admin: deque[Callable[[], None]] = deque()
        self._visitors: deque[Job] = deque()
        self._loaded: _Loaded | None = None
        self._status = "loading"
        self._model_key = initial_model
        self._faders = _Faders()
        self._stopping = False
        self.poller = Poller(self._publish_state)
        self._thread = threading.Thread(target=self._run, name="steerlab-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        with self._wake:
            self._stopping = True
            self._wake.notify_all()
        self.poller.cancel("reset")
        self._thread.join(timeout)

    def _run(self) -> None:
        self._swap_to(self._model_key, previous=None)
        while True:
            with self._wake:
                while not (self._stopping or self._admin or self._visitors):
                    self._wake.wait()
                if self._stopping:
                    break
                command = self._admin.popleft() if self._admin else None
                job = None if command is not None else self._visitors.popleft()
                if job is not None:
                    self._send_positions()
            try:
                self._apply_pending()
                if command is not None:
                    command()
                elif job is not None:
                    self._reply(job)
            except Exception as error:  # the only thread that runs the model must outlive any one failure
                self.notice(f"Something failed and was skipped: {error}", "error")
                if job is not None:
                    self.store.release(job.conversation)
                    job.sink({"type": "error", "message": UNAVAILABLE})
                    job.sink({"type": "end", "reason": "failed"})
        self._close_model()

    def submit(self, conversation: str, text: str, sink: Sink) -> Job:
        """Queue a visitor message. Raises MessageRejected (empty, too long, one already pending, queue full)."""
        self.store.reserve(conversation, text)
        job = Job(conversation, text, sink)
        with self._wake:
            if len(self._visitors) >= QUEUE_CAPACITY:
                self.store.release(conversation)
                raise QueueFull("The assistant is busy. Try again in a minute.")
            self._visitors.append(job)
            position = len(self._visitors) + (1 if self.poller.active else 0)
            self._wake.notify_all()
        sink({"type": "queued", "position": position})
        self._publish_state()
        return job

    def disconnect(self, job: Job) -> None:
        """The visitor's tab went away: drop the queued job, or stop its reply within one token."""
        with self._wake:
            if job in self._visitors:
                self._visitors.remove(job)
                job.cancelled = True
                self.store.release(job.conversation)
                return
        self.poller.cancel("disconnected", job.conversation)

    def set_strength(self, name: str, strength: float) -> None:
        """Move one fader: a direction name, or `feature:<id>` for an SAE feature the admin added."""
        if not math.isfinite(strength):
            raise ValueError("strength must be a finite number")
        with self._lock:
            if name.startswith("feature:"):
                feature = self._feature_id(name)
                if feature not in self._faders.features:
                    raise ValueError(f"feature {feature} is not on the desk; add it first")
                self._faders.features[feature] = strength
            else:
                if name not in self._direction_names():
                    raise ValueError(f"unknown direction {name!r}")
                self._faders.directions[name] = strength
            self._push_control()

    def all_off(self) -> None:
        with self._lock:
            self._faders.directions = dict.fromkeys(self._faders.directions, 0.0)
            self._faders.features = dict.fromkeys(self._faders.features, 0.0)
            self._push_control()

    def add_feature(self, feature: int) -> None:
        """Put one SAE feature on the desk at strength 0, with its label when one is cached."""
        with self._lock:
            loaded = self._loaded
            if loaded is None or loaded.entry.sae is None or loaded.n_features is None:
                raise ValueError("this model has no SAE loaded, so it has no features to add")
            if not 0 <= feature < loaded.n_features:
                raise ValueError(f"feature ids run from 0 to {loaded.n_features - 1}")
            if feature in self._faders.features:
                return
            if len(self._faders.features) >= MAX_FEATURES:
                raise ValueError(f"at most {MAX_FEATURES} features at a time; remove one first")
            self._faders.features[feature] = 0.0
            label = None
            if self._feature_label is not None:
                sae = loaded.entry.sae
                try:
                    label = self._feature_label(loaded.entry.model_id, sae.release, sae.layer, feature)
                except Exception:  # a broken label cache must not block steering
                    label = None
            if label:
                self._faders.labels[feature] = label
            self._push_control()

    def remove_feature(self, feature: int) -> None:
        with self._lock:
            if self._faders.features.pop(feature, None) is None:
                raise ValueError(f"feature {feature} is not on the desk")
            self._faders.labels.pop(feature, None)
            self._faders.modes.pop(feature, None)
            self._push_control()

    def stop_reply(self) -> bool:
        """End the reply being written now, keeping what was written (the admin's Stop)."""
        return self.poller.cancel("stopped")

    def set_prompt(self, text: str | None) -> None:
        """Replace the system prompt from the next reply on (`None`: back to the launch default)."""
        prompt = self._default_prompt if text is None else text.strip()
        self._admin_command(lambda: self._set_prompt(prompt))

    def redo(self, conversation: str) -> None:
        self._admin_command(lambda: self._redo(conversation))

    def clear(self) -> None:
        """Empty every conversation and reset the faders (cancels the reply in progress)."""
        self.poller.cancel("reset")
        self.poller.discard()
        self._admin_command(self._clear)

    def swap(self, key: str) -> None:
        """Load another catalogued model. Raises ValueError for a key outside the catalog (no download, no path)."""
        if self._entry(key, refresh=True) is None:
            raise ValueError(f"{key!r} is not a downloaded model; choose one from the model list")
        self.poller.cancel("reset")
        self.poller.discard()
        self._admin_command(lambda: self._swap(key))

    def new_conversation(self, conversation: str) -> None:
        """The visitor's "new conversation": stop its reply if one is running, then forget its turns."""
        self.poller.cancel("reset", conversation)
        self._admin_command(lambda: self._new_conversation(conversation))

    def status(self, conversation: str) -> dict[str, Any]:
        """What an idle visitor tab polls: a neutral availability word and its conversation's version."""
        with self._lock:
            available = "ready" if self._status == "ready" else "restarting"
        return {"status": available, "version": self.store.version(conversation)}

    def state(self) -> Event:
        with self._lock:
            return self._state_event()

    def _admin_command(self, command: Callable[[], None]) -> None:
        with self._wake:
            self._admin.append(command)
            self._wake.notify_all()

    def _entry(self, key: str, *, refresh: bool = False) -> CatalogEntry | None:
        """The catalog entry for `key`. The catalog scans the Hugging Face cache on disk, so it is re-read only when a
        swap is requested (a model downloaded meanwhile then appears), not on every state event."""
        if refresh:
            models = self._catalog()
            with self._lock:
                self._models = models
        return next((entry for entry in self._models if entry.key == key), None)

    def _wake_worker(self) -> None:
        """An empty command: the idle worker wakes, applies pending fader changes at the boundary, and sleeps again."""
        self._admin_command(lambda: None)

    def _apply_pending(self) -> None:
        with contextlib.suppress(ReplyCancelled):
            self.poller.poll()

    def _feature_id(self, name: str) -> int:
        value = name.partition(":")[2]
        if not value.isdigit():
            raise ValueError(f"{name!r}: a feature is feature:<whole number>")
        return int(value)

    def _direction_names(self) -> list[str]:
        loaded = self._loaded
        return [] if loaded is None else sorted(getattr(loaded.state.resolver, "directions", {}))

    def _control_file(self) -> ControlFile | None:
        loaded = self._loaded
        if loaded is None:
            return None
        entries = [
            ControlEntry(source=DirectionSource(kind="direction", name=name), strength=strength)
            for name, strength in self._faders.directions.items()
            if strength != 0
        ]
        sae = loaded.entry.sae
        if sae is not None:
            entries += [
                ControlEntry(
                    source=SAESource(kind="sae", release=sae.release, layer=sae.layer, feature=feature),
                    strength=strength,
                    mode="clamp" if self._faders.modes.get(feature) == "clamp" else None,
                )
                for feature, strength in self._faders.features.items()
                if strength != 0
            ]
        return ControlFile(model=loaded.state.model, entries=entries)

    def _push_control(self) -> None:
        """Hand the faders to the poller: a running reply takes them before its next token, an idle worker at once."""
        control = self._control_file()
        if control is None:
            return
        self.poller.push(control)
        if self.poller.active is None:
            self._wake_worker()

    def _state_event(self) -> Event:
        loaded = self._loaded
        directions = []
        features = []
        sae = None
        if loaded is not None:
            registry: Any = loaded.state.resolver
            preset = self._presets.get(loaded.entry.key)
            suggest = preset.suggest if preset is not None else {}
            for name in self._direction_names():
                direction = registry.directions[name]
                directions.append(
                    {
                        "name": name,
                        "label": direction.label or name,
                        "layer": direction.layer,
                        "strength": self._faders.directions.get(name, 0.0),
                        "suggested": suggest.get(name),
                    }
                )
            if loaded.entry.sae is not None and loaded.n_features is not None:
                sae = {
                    "release": loaded.entry.sae.release,
                    "layer": loaded.entry.sae.layer,
                    "features": loaded.n_features,
                }
                feature_suggest = {item.id: item.suggest for item in preset.features} if preset is not None else {}
                features = [
                    {
                        "id": feature,
                        "label": self._faders.labels.get(feature),
                        "strength": strength,
                        "suggested": feature_suggest.get(feature),
                        "mode": self._faders.modes.get(feature, "add"),
                    }
                    for feature, strength in self._faders.features.items()
                ]
        return {
            "type": "state",
            "status": self._status,
            "model": self._model_key,
            "model_id": None if loaded is None else loaded.state.model,
            "models": [entry.to_dict() for entry in self._models],
            "prompt": self._prompt,
            "default_prompt": self._default_prompt,
            "directions": directions,
            "features": features,
            "sae": sae,
            "max_features": MAX_FEATURES,
            "queue": len(self._visitors),
            "active": self.poller.active,
        }

    def _publish_state(self) -> None:
        with self._lock:
            event = self._state_event()
        self.broker.publish(event)

    def notice(self, message: str, level: str = "info") -> None:
        """Show a message in every admin tab's notices."""
        self.broker.publish({"type": "notice", "level": level, "message": message})

    def _send_positions(self) -> None:
        for position, job in enumerate(self._visitors, start=2):  # position 1 is the reply about to start
            job.sink({"type": "queued", "position": position})

    def _set_prompt(self, prompt: str) -> None:
        with self._lock:
            self._prompt = prompt
        self._publish_state()
        self.notice("Prompt applied: it is used from the next reply.")

    def _reset_everything(self) -> None:
        """Clear and swap: every queued visitor gets `end: reset`; conversations and faders reset; mirror cleared."""
        self.poller.discard()
        with self._wake:
            jobs, self._visitors = list(self._visitors), deque()
            self._faders = _Faders()
            if self._loaded is not None:
                self._loaded.state.clear()
        for job in jobs:
            job.cancelled = True
            job.sink({"type": "end", "reason": "reset"})
        self.store.clear_all()
        self.broker.publish({"type": "reset"})

    def _apply_preset(self, key: str) -> None:
        """The loaded model's preset: its own prompt (when prompts follow the model) and its features on the desk."""
        if self._prompt_for is not None:
            prompt = self._prompt_for(key)
            with self._lock:
                self._prompt = self._default_prompt = prompt
        preset = self._presets.get(key)
        loaded = self._loaded
        if preset is None or loaded is None or loaded.n_features is None:
            return
        with self._lock:
            for item in preset.features:
                if 0 <= item.id < loaded.n_features and len(self._faders.features) < MAX_FEATURES:
                    self._faders.features.setdefault(item.id, 0.0)
                    self._faders.labels[item.id] = item.label
                    if item.mode == "clamp":
                        self._faders.modes[item.id] = "clamp"

    def _clear(self) -> None:
        self._reset_everything()
        if self._loaded is not None:
            self._apply_preset(self._loaded.entry.key)
        self._publish_state()
        self.notice("Cleared every conversation and set every fader to 0.")

    def _new_conversation(self, conversation: str) -> None:
        self.store.reset(conversation)
        self.broker.publish({"type": "new", "conversation": conversation})

    def _swap(self, key: str) -> None:
        if self._entry(key) is None:
            self.notice(f"{key!r} is not a downloaded model; nothing changed", "error")
            return
        self._reset_everything()
        self._swap_to(key, previous=self._model_key)

    def _close_model(self) -> None:
        loaded, self._loaded = self._loaded, None
        self.poller.state = None
        if loaded is not None:
            loaded.session.__exit__(None, None, None)
            self._unload(loaded.backend, loaded.state)

    def _open(self, key: str) -> None:
        from steerlab.engine import SteeringSession

        entry = self._entry(key)
        if entry is None:
            raise ValueError(f"{key!r} is not a downloaded model")
        backend, state = self._load(entry, self.notice)
        session = SteeringSession(backend.model, backend.tokenizer, state)
        session.__enter__()
        saes: dict[tuple[str, int], Any] = getattr(state.resolver, "saes", {})
        n_features = None
        if entry.sae is not None and (entry.sae.release, entry.sae.layer) in saes:
            n_features = int(saes[(entry.sae.release, entry.sae.layer)].n_features)
        self._loaded = _Loaded(entry, backend, state, session, n_features)
        self.poller.state = state
        self._warm_up(session)
        self._apply_preset(key)

    def _warm_up(self, session: SteeringSession) -> None:
        """One forward pass, so the first visitor does not pay for kernel compilation."""
        import torch

        ids = session.tokenizer("Hello", return_tensors="pt", add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            session.model(input_ids=ids.to(session.device))

    def _swap_to(self, key: str, *, previous: str | None) -> None:
        self._set_status("loading" if previous is None else "swapping", key)
        self._close_model()
        failure = None
        try:
            self._open(key)
        except Exception as error:
            failure = str(error)  # keep only the text: the exception's frames would hold the failed weights
        if failure is None:
            self._set_status("ready", key)
            self.notice(f"Loaded {key}.")
            return
        self.notice(f"Could not load {key}: {failure}", "error")
        self._close_model()
        gc.collect()
        if previous is None:
            self._set_status("failed", key)
            return
        self._set_status("restoring", previous)
        try:
            self._open(previous)
        except Exception as again:
            self.notice(f"Could not load {previous} again: {again}", "error")
            self._close_model()
            self._set_status("failed", previous)
            return
        self._set_status("ready", previous)

    def _set_status(self, status: str, key: str) -> None:
        with self._lock:
            self._status, self._model_key = status, key
        self._publish_state()

    def _reply(self, job: Job) -> None:
        if job.cancelled:
            return
        if self._loaded is None:
            self.store.release(job.conversation)
            job.sink({"type": "error", "message": UNAVAILABLE})
            job.sink({"type": "end", "reason": "failed"})
            return
        history = self.store.begin_turn(job.conversation, job.text)
        self.broker.publish({"type": "user", "conversation": job.conversation, "text": job.text})
        self._generate(job.conversation, history, job.sink, finish=self.store.finish_turn, redo=False)

    def _redo(self, conversation: str) -> None:
        if self._loaded is None:
            self.notice("No model is loaded.", "error")
            return
        history = self.store.begin_redo(conversation)
        if history is None:
            self.notice("Nothing to redo in that conversation yet.", "error")
            return
        self.broker.publish({"type": "redo", "conversation": conversation})
        self._generate(conversation, history, None, finish=self.store.finish_redo, redo=True)

    def _generate(
        self,
        conversation: str,
        history: list[Message],
        sink: Sink | None,
        *,
        finish: Callable[[str, str | None], None],
        redo: bool,
    ) -> None:
        """Write one reply into `conversation`, then `finish(conversation, text or None)` before the `end` event goes
        out, so a visitor who sends again on `end` finds the turn already stored."""
        from steerlab.decode import encode_prompt, greedy_generate, render_messages

        assert self._loaded is not None
        session = self._loaded.session
        with self._lock:
            prompt = self._prompt
        messages = ([{"role": "system", "content": prompt}] if prompt else []) + history

        def send(event: Event) -> None:
            self.broker.publish(event)
            if sink is not None:
                sink(event)

        try:
            text = render_messages(session.tokenizer, messages, model_type=session.model_type)
            ids = encode_prompt(session.tokenizer, text, session.device)
        except Exception as error:  # e.g. a chat template that refuses this history
            self.notice(f"The prompt could not be built: {error}", "error")
            if sink is not None:
                sink({"type": "error", "message": UNAVAILABLE})
            send({"type": "start", "conversation": conversation})
            finish(conversation, None)
            send({"type": "end", "conversation": conversation, "reason": "failed", "dropped": True, "keep_user": redo})
            return

        if int(ids.shape[1]) > MAX_PROMPT_TOKENS:
            if sink is not None:
                sink({"type": "error", "message": FULL_MESSAGE})
            send({"type": "start", "conversation": conversation})
            finish(conversation, None)
            send({"type": "end", "conversation": conversation, "reason": "full", "dropped": True, "keep_user": redo})
            return
        pieces: list[str] = []

        def on_token(index: int, piece: str) -> None:
            pieces.append(piece)
            send({"type": "token", "conversation": conversation, "index": index, "text": piece})

        def on_change(entry: TimelineEntry) -> None:
            self.broker.publish(
                {
                    "type": "marker",
                    "conversation": conversation,
                    "token_index": entry.token_index,
                    "applied": [applied.to_dict() for applied in entry.applied],
                    "added_norm": {str(layer): norm for layer, norm in entry.added_norm.items()},
                }
            )

        send({"type": "start", "conversation": conversation, "model": self._model_key})
        self.poller.begin(conversation)
        self._publish_state()
        reply: str | None
        try:
            generation = greedy_generate(
                session,
                ids,
                max_new_tokens=self.max_new_tokens,
                watcher=self.poller,
                on_token=on_token,
                on_change=on_change,
            )
            reason = "eos" if generation.finish_reason == "stop" else "length"
            reply = generation.text
        except ReplyCancelled as cancelled:
            reason = cancelled.reason
            reply = "".join(pieces) if reason == "stopped" else None
        except Exception as error:  # a failed reply must not kill the worker
            self.notice(f"The reply failed: {error}", "error")
            if sink is not None:
                sink({"type": "error", "message": UNAVAILABLE})
            reason, reply = "failed", None
        finally:
            self.poller.end()
        finish(conversation, reply)
        dropped = reply is None
        send({"type": "end", "conversation": conversation, "reason": reason, "dropped": dropped, "keep_user": redo})
        self._publish_state()
