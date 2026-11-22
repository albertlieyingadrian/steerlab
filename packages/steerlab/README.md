# steerlab

Live activation steering on small open models. steerlab adds a vector to a model's residual stream while it writes,
one token at a time, and lets you change that vector mid-reply.

- **The control file** (`steerlab.control`): a small JSON file that says what is being steered right now: which
  directions or SAE features, at what strength, added or clamped. It is re-read before every token, and a half-written
  file never breaks a running reply: the last valid state stays.
- **The engine** (`steerlab.engine`): forward hooks that add the control state's vectors at generated positions only.
  The prompt is never steered, and strength 0 is exactly the unsteered model.
- **The decode loop** (`steerlab.decode`): one forward pass per token, checking the control file before each one, so
  a change applies from the next token.
- **Directions** (`steerlab.directions`, `steerlab directions build|list`): difference-of-means "arrows" from pairs of
  example replies (in Chinese vs in English, refusing vs answering, ...), saved with the layer they were read at and
  three quality numbers: the class gap, how much the pairs agree (cosine agreement), and how well the arrow separates
  the two sets (d′). Saved under `$STEERLAB_DIRECTIONS`, else `outputs/directions`.

- **The terminal** (`steerlab chat`, `steerlab control set|off`, `steerlab doctor`): chat with a model in one
  terminal and steer it from another while it writes. `doctor` reports the device, what is cached and which
  directions exist, offline, and never prints a token.
- **The inspector** (`steerlab inspect`, then open http://127.0.0.1:8770): a local web app in rooms. **Trace** splits
  the next word's score into one share per layer, head and neuron and lets you switch neurons off; **Lens** reads
  every layer and position as words through the logit, J, R and S lenses; **Units** is an atlas of every neuron pole
  and head (labels, top contexts, token drivers, reads, wiring), from the neuron-lens catalog or computed; **Edit** makes
  concept edits in the weights (mirror, project out, amplify) and undoes them exactly; **Workbench** chats with the
  model under steering, with probes, branching replies and saved artifacts; **Model** shows the loaded model's shape.

Strength is measured in *class gaps*: strength 1.0 moves the residual stream as far along a direction as the average
gap between the two example sets the direction was built from.

## Install

```bash
pip install "steerlab[models]"    # torch, transformers and the SAE loaders; the base install never imports torch
```

In this repo, `uv sync --all-packages` installs it with sandbagkit, editable.
