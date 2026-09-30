"""Collect Google search hit counts for every paper in the prevalence-proxy set.

For each paper the query is the quoted title followed by the quoted author names,
e.g. ``"Attention is all you need" "Ashish Vaswani" "Noam Shazeer" ...`` (the
``questions`` column built in step 01/04). The total number of results reported
by Google is retrieved through BrightData's SERP API.

Papers are processed in batches of 500; every batch is saved to its own CSV so
an interrupted run can be resumed (finished batches are skipped). Queries that
fail (timeouts, empty result pages, ...) are skipped, which is why fewer than
6,048 papers end up with a hit count (5,179 in the paper). Finally all batch
files are merged into one CSV.

Requires BRIGHTDATA_API_KEY and BRIGHTDATA_ZONE in the environment or ``.env``.
BrightData bills per request; the crawl stops early if the account balance runs out.

Input:
    config.PROXY_SET_CSV
Output:
    config.GOOGLE_HITS_DIR / crawl_batches / google_results_batch_{start}_{end}.csv
    config.GOOGLE_HITS_DIR / combined_google_results.csv
        columns: question, citation, number of hits, time taken

Usage:
    python 02_google_search_hits/crawl_google_hits.py [--batch-size 500]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

BRIGHTDATA_URL = "https://api.brightdata.com/request"
BATCH_DIR = config.GOOGLE_HITS_DIR / "crawl_batches"
COMBINED_CSV = config.GOOGLE_HITS_DIR / "combined_google_results.csv"


def get_google_hits(query: str, api_key: str, zone: str) -> tuple[float | None, float | None]:
    """Return Google's result count and search time for one query.

    Args:
        query: The search string.
        api_key: BrightData API key.
        zone: BrightData SERP zone name.

    Returns:
        ``(results_count, search_time)``, or ``(None, None)`` if the request failed.
    """
    url = f"https://www.google.com/search?q={quote_plus(query)}&brd_json=1"
    payload = {"zone": zone, "url": url, "format": "json"}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        response = requests.post(BRIGHTDATA_URL, headers=headers, data=json.dumps(payload), timeout=60)
        body_str = response.json().get("body")
        if not body_str:
            return None, None
        general = json.loads(body_str).get("general")
    except (requests.RequestException, json.JSONDecodeError, AttributeError):
        return None, None
    if not general:
        return None, None
    return general.get("results_cnt"), general.get("search_time")


def crawl(papers: pd.DataFrame, api_key: str, zone: str, batch_size: int) -> None:
    """Query hit counts batch by batch, skipping batches that were already saved.

    Args:
        papers: DataFrame with ``questions`` and ``citationCount`` columns.
        api_key: BrightData API key.
        zone: BrightData SERP zone name.
        batch_size: Number of papers per saved batch.
    """
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(papers), batch_size):
        end = min(start + batch_size, len(papers))
        out_path = BATCH_DIR / f"google_results_batch_{start}_{end - 1}.csv"
        if out_path.exists():
            print(f"Skipping finished batch {out_path.name}")
            continue

        results = []
        for _, row in tqdm(papers.iloc[start:end].iterrows(), total=end - start, desc=f"Batch {start}-{end - 1}"):
            question = str(row["questions"])
            num_hits, time_taken = get_google_hits(question, api_key, zone)
            if num_hits is None or time_taken is None:
                print(f"Skipped: {question}")
                continue
            results.append([question, row["citationCount"], float(num_hits), float(time_taken)])

        pd.DataFrame(results, columns=["question", "citation", "number of hits", "time taken"]).to_csv(
            out_path, index=False, encoding="utf-8"
        )
        print(f"Saved {len(results)} results to {out_path}")


def merge_batches() -> None:
    """Concatenate all batch files into ``combined_google_results.csv``."""
    batch_files = sorted(BATCH_DIR.glob("google_results_batch_*.csv"), key=lambda p: int(p.stem.split("_")[3]))
    combined = pd.concat([pd.read_csv(f) for f in batch_files], ignore_index=True)
    combined.to_csv(COMBINED_CSV, index=False)
    print(f"Merged {len(batch_files)} batches ({len(combined)} papers) into {COMBINED_CSV}")


def main() -> None:
    """Parse arguments, crawl all batches, and merge them."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    api_key = os.environ.get("BRIGHTDATA_API_KEY", "").strip()
    zone = os.environ.get("BRIGHTDATA_ZONE", "").strip()
    if not api_key or not zone:
        sys.exit("ERROR: set BRIGHTDATA_API_KEY and BRIGHTDATA_ZONE (see .env.example).")

    papers = pd.read_csv(config.PROXY_SET_CSV)[["questions", "citationCount"]]
    crawl(papers, api_key, zone, args.batch_size)
    merge_batches()


if __name__ == "__main__":
    main()
