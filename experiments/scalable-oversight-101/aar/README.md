# A small automated weak-to-strong researcher

The rules are in [`program.md`](program.md). `prepare.py` is fixed; `method.py` is the only file a run changes; `run.py --note "..."` scores it on dev and logs it; `final_test.py RUN ...` scores named runs once on test.

`results.tsv` has one row per run: `run`, `split` (`dev`, or `test` for the final rows), `acc`, `pgr`, `seconds`, `note`. `runs/NNN_method.py` is the exact `method.py` of each run.
