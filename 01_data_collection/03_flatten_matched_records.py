"""Step 3: flatten the Semantic Scholar-matched NDJSON into CSV chunks of 10,000 papers.

Each NDJSON record holds the original OAG fields under ``original.*`` and the
matched Semantic Scholar record under ``semantic_scholar.*``. Nested fields are
flattened with ``pd.json_normalize`` (e.g. ``semantic_scholar.citationCount``).

Input:
    config.MATCHED_NDJSON
Output:
    config.MATCHED_CHUNKS_DIR / chunk_{0..N}.csv

Usage:
    python 01_data_collection/03_flatten_matched_records.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

CHUNK_SIZE = 10_000


def write_chunk(records: list[dict], chunk_id: int) -> None:
    """Flatten a list of records and save it as ``chunk_{chunk_id}.csv``.

    Args:
        records: Parsed NDJSON records.
        chunk_id: Index of the output chunk.
    """
    out_path = config.MATCHED_CHUNKS_DIR / f"chunk_{chunk_id}.csv"
    pd.json_normalize(records).to_csv(out_path, index=False)
    print(f"Saved {out_path}")


def main() -> None:
    """Stream the NDJSON file and write flattened CSV chunks."""
    config.MATCHED_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    buffer: list[dict] = []
    chunk_id = 0
    with open(config.MATCHED_NDJSON, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(tqdm(f, desc="Reading NDJSON")):
            try:
                buffer.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON at line {line_num}")
                continue
            if len(buffer) >= CHUNK_SIZE:
                write_chunk(buffer, chunk_id)
                buffer = []
                chunk_id += 1
    if buffer:
        write_chunk(buffer, chunk_id)


if __name__ == "__main__":
    main()
