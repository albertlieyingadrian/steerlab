Copies of the base 2D world-model project's simulator (`physics.py`), world model (`worldmodel.py`) and probing
helpers (`probing.py`), so this experiment runs on its own. The code's only behaviour change is
`load_checkpoint`'s error message in `worldmodel.py`, which names this experiment's training scripts; comments and
docstrings are reworded to stand alone. `probing.py` imports `sandbagkit` from `packages/`.
