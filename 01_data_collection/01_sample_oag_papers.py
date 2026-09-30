"""Step 1: randomly sample 10 million papers from the Open Academic Graph v2 (MAG part).

The OAG v2 MAG dump is split into ``mag_papers_0.txt`` ... ``mag_papers_10.txt``
(JSON lines, one paper per line). Holding 10M papers in memory at once is not
practical, so the sample is drawn in 10 rounds of 1M papers each. Every round
runs reservoir sampling over the full dump while skipping papers already drawn
in earlier rounds, so the 10 chunks are disjoint.

Input:
    config.OAG_RAW_DIR / mag_papers_{0..10}.txt
Output:
    config.OAG_SAMPLE_DIR / sampled_mag_papers_all_fields_chunk_{0..9}.csv

Usage:
    python 01_data_collection/01_sample_oag_papers.py
"""

import json
import random
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

RANDOM_SEED = 1234
SAMPLE_SIZE = 10_000_000
BATCH_SIZE = 1_000_000


def reservoir_sample_round(excluded_ids: set[str]) -> list[dict]:
    """Draw one uniform sample of ``BATCH_SIZE`` papers with reservoir sampling.

    Args:
        excluded_ids: Paper IDs drawn in previous rounds; they are skipped.

    Returns:
        The sampled paper records (raw JSON dictionaries).
    """
    total_seen = 0
    reservoir: list[dict] = []
    for i in tqdm(range(config.OAG_FILE_COUNT), desc="Reservoir sampling"):
        file_path = config.OAG_RAW_DIR / f"{config.OAG_FILE_PREFIX}{i}.txt"
        if not file_path.exists():
            print(f"Missing file, skipped: {file_path}")
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=file_path.name, leave=False):
                try:
                    paper = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = paper.get("id")
                if pid is None or pid in excluded_ids:
                    continue
                total_seen += 1
                if len(reservoir) < BATCH_SIZE:
                    reservoir.append(paper)
                else:
                    r = random.randint(0, total_seen - 1)
                    if r < BATCH_SIZE:
                        reservoir[r] = paper
    return reservoir


def flatten_papers(papers: list[dict]) -> pd.DataFrame:
    """Flatten nested OAG fields so the sample can be stored as CSV.

    ``authors`` becomes a "; "-joined string of names and ``venue`` becomes its
    raw venue string.

    Args:
        papers: Raw OAG paper records.

    Returns:
        A DataFrame with one row per paper and one column per OAG field.
    """
    all_keys: set[str] = set()
    for p in papers:
        all_keys.update(p.keys())
    columns = sorted(all_keys)

    rows = []
    for p in papers:
        row = {k: p.get(k) for k in columns}
        if isinstance(row.get("authors"), list):
            row["authors"] = "; ".join(a.get("name", "") for a in row["authors"] if isinstance(a, dict))
        if isinstance(row.get("venue"), dict):
            row["venue"] = row["venue"].get("raw")
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def main() -> None:
    """Run all sampling rounds and write one CSV chunk per round."""
    random.seed(RANDOM_SEED)
    config.OAG_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    sampled_ids: set[str] = set()
    for k in range(SAMPLE_SIZE // BATCH_SIZE):
        print(f"---------- Sampling round {k + 1}/{SAMPLE_SIZE // BATCH_SIZE} ----------")
        reservoir = reservoir_sample_round(sampled_ids)
        sampled_ids.update(p["id"] for p in reservoir if p.get("id"))

        out_path = config.OAG_SAMPLE_DIR / f"sampled_mag_papers_all_fields_chunk_{k}.csv"
        flatten_papers(reservoir).to_csv(out_path, index=False)
        print(f"Saved {len(reservoir)} papers to {out_path}")


if __name__ == "__main__":
    main()
