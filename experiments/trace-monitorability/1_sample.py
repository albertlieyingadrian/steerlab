"""
Step 1: draw 120 trajectories from nebius/SWE-agent-trajectories, stratified 20 per (model, resolved) cell.

No full download: each row comes from the Hugging Face datasets-server /filter endpoint, one at a time, at a seeded
random offset within its cell. Rows longer than MAX_TURNS entries or with fewer than 6 agent steps are skipped and
another offset is drawn. Raw rows go to WORK/raw/<row_idx>.json; results/episodes.csv holds IDs and labels only.
"""

from common import MAX_TURNS, MODELS, PER_CELL, RESULTS, WORK, cell_size, check, fetch_row, load_raw, rng, write_csv

r = rng()
rows = []
for model in MODELS:
    for resolved in (True, False):
        n = cell_size(model, resolved)
        offsets = r.sample(range(n), min(n, 400))
        got, tried, skipped = 0, 0, 0
        for off in offsets:
            if got == PER_CELL:
                break
            tried += 1
            row = fetch_row(model, resolved, off)
            n_ai = sum(t["role"] == "ai" for t in row["trajectory"])
            if len(row["trajectory"]) > MAX_TURNS or n_ai < 6:
                skipped += 1
                (WORK / "raw" / f"{row['row_idx']}.json").unlink()
                continue
            got += 1
            rows.append(
                {
                    "episode": f"swe:{row['row_idx']}",
                    "row_idx": row["row_idx"],
                    "instance_id": row["instance_id"],
                    "model_name": model,
                    "resolved": int(row["target"]),
                    "exit_status": row["exit_status"],
                    "n_entries": len(row["trajectory"]),
                    "n_ai": n_ai,
                    "cell_offset": off,
                }
            )
        print(f"{model:24s} resolved={resolved!s:5s} cell size {n:6d}: kept {got}, skipped {skipped} (too long/short)")

write_csv(RESULTS / "episodes.csv", rows)
check("120 episodes", len(rows) == 3 * 2 * PER_CELL)
check("all raw rows cached", all(load_raw(x["row_idx"]) for x in rows))
check("no duplicate rows", len({x["row_idx"] for x in rows}) == len(rows))
print("wrote results/episodes.csv")
