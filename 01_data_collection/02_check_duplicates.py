"""Step 2 (sanity check): verify that the 10 sampled OAG chunks contain no duplicate papers.

Input:
    config.OAG_SAMPLE_DIR / sampled_mag_papers_all_fields_chunk_{0..9}.csv

Usage:
    python 01_data_collection/02_check_duplicates.py
"""

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

NUM_CHUNKS = 10


def main() -> None:
    """Count papers and unique paper IDs across all chunks."""
    unique_ids: set[str] = set()
    paper_count = 0
    for i in tqdm(range(NUM_CHUNKS), desc="Loading chunks"):
        path = config.OAG_SAMPLE_DIR / f"sampled_mag_papers_all_fields_chunk_{i}.csv"
        chunk = pd.read_csv(path, usecols=["id"], dtype={"id": str})
        paper_count += len(chunk)
        unique_ids.update(chunk["id"])

    print(f"Papers: {paper_count:,}  unique IDs: {len(unique_ids):,}")
    if len(unique_ids) == paper_count:
        print("OK: no duplicates.")
    else:
        print(f"WARNING: {paper_count - len(unique_ids):,} duplicate papers found.")


if __name__ == "__main__":
    main()
