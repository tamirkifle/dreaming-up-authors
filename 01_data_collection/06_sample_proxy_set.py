"""Step 6: build the prevalence-proxy set (up to 500 papers per citation bin, 6,048 papers).

This set is used for the Google search hits analysis (module 02) and the
Infini-gram corpus occurrence analysis (module 03). Chunks are scanned in file
order and each paper is kept while its citation bin holds fewer than 500
papers. The two highest bins cannot be filled.

With ``--resample`` the chunks are scanned in reverse order, which gives an
independent second sample (used for the Infini-gram robustness check).

Input:
    config.QUERY_FIELDS_DIR / samples_google_search_{i}.csv
Output:
    config.PROXY_SET_CSV          (default)
    config.PROXY_RESAMPLE_CSV     (--resample)

Usage:
    python 01_data_collection/06_sample_proxy_set.py [--resample]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

MAX_PER_BIN = 500


def sample_bins(reverse: bool) -> pd.DataFrame:
    """Fill every citation bin greedily while scanning the query-field chunks.

    Args:
        reverse: Scan the chunks from last to first instead of first to last.

    Returns:
        The selected papers.
    """
    labels = config.CITATION_BIN_LABELS
    target_total = MAX_PER_BIN * len(labels)
    bin_rows: dict[str, list[pd.Series]] = {label: [] for label in labels}
    bin_counts = {label: 0 for label in labels}

    chunk_paths = sorted(
        config.QUERY_FIELDS_DIR.glob("samples_google_search_*.csv"),
        key=lambda p: int(p.stem.rsplit("_", 1)[1]),
        reverse=reverse,
    )
    for path in tqdm(chunk_paths, desc="Scanning chunks"):
        data = pd.read_csv(path)
        data["citation_bin"] = pd.cut(data["citationCount"], bins=config.CITATION_BIN_EDGES, labels=labels)
        for _, row in data.iterrows():
            bin_label = row["citation_bin"]
            if bin_counts[bin_label] >= MAX_PER_BIN:
                continue
            bin_rows[bin_label].append(row)
            bin_counts[bin_label] += 1
            if sum(bin_counts.values()) >= target_total:
                break
        if sum(bin_counts.values()) >= target_total:
            break

    return pd.concat([pd.DataFrame(bin_rows[label]) for label in labels], ignore_index=True)


def main() -> None:
    """Sample the proxy set and save it."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resample", action="store_true", help="scan chunks in reverse order")
    args = parser.parse_args()

    df = sample_bins(reverse=args.resample)
    df = df.dropna(subset=["year"]).dropna(subset=["citationCount"])
    print(f"Proxy set: {len(df)} papers")
    print(df["citation_bin"].value_counts(sort=False).to_string())

    out_path = config.PROXY_RESAMPLE_CSV if args.resample else config.PROXY_SET_CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
