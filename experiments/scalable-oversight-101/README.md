# Scalable oversight 101: weak-to-strong and sandwiching on a laptop

Research note: [`report/index.html`](report/index.html). Status: introduction published; experiments in progress.

- `make_cover.py` draws the cover (`report/figures/hero.jpg`) and the research-page thumbnail.

The plan:

- **Weak-to-strong:** Qwen2.5-0.5B labels GSM8K-verification items, and Qwen2.5-1.5B learns from those labels. The score is the performance gap recovered, compared across four simple methods.
- **Sandwiching:** a 0.5B judge decides alone, after one argument from a 7B model, or after a debate between two arguments.
- **A small automated researcher:** tuned on dev, and reported once on test.
