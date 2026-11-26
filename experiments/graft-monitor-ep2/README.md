# graft-monitor-ep2

The Episode 2 note, "Does alignment survive self-improvement?" (`report/index.html`). Its code, data and results live
with Episode 1 in [`../graft-monitor/`](../graft-monitor/): the generations are trained by `trainer/modal_train.py`
(`launch_ep2.sh`), scored by `analyze_ep2.py`, the values battery is in `battery/`, the self-monitor test is
`q3_selfmon.py`, and `make_figures_ep2.py` draws this note's figures into `report/figures/` here.
`bash ../graft-monitor/reproduce.sh --dry-run` prints every command behind both notes.
