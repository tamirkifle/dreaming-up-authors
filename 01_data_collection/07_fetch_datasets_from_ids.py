"""Step 7 (shortcut): rebuild the paper sets used in the paper from their Semantic Scholar IDs.

Steps 1-6 need the full OAG dump and the Semantic Scholar matching output. To
run modules 02-05 on exactly the same papers as the paper, this script instead
downloads titles, years and authors from the Semantic Scholar Graph API
(``/paper/batch``) for the papers listed in the released ID files, and writes
the same files as steps 5 and 6:

    --set eval             paper_ids_9108.csv            -> config.EVAL_SET_CSV        (modules 04, 05)
    --set proxy            paper_ids_6048.csv            -> config.PROXY_SET_CSV       (modules 02, 03)
    --set proxy_resample   paper_ids_6048_resample.csv   -> config.PROXY_RESAMPLE_CSV  (module 03)
    --set all              all three

The ID files also store each paper's citation count and citation bin from the
September 2025 snapshot used in the paper; the evaluation-set file additionally
stores the field and the author list (``authors-info``: names and Semantic
Scholar author IDs). These stored values are written to the output, so every
paper keeps its original bin, field and ground-truth authors even though
Semantic Scholar revises counts and author names over time. Only titles and
years (and, for the proxy sets, authors) are fetched live. The script reports
how many papers would fall into a different bin today.

The row order of the ID files is preserved, which matters because the
multiple-choice option builder (module 05) consumes its random number
generator in row order.

Usage:
    python 01_data_collection/07_fetch_datasets_from_ids.py --set all
"""

import argparse
import ast
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_FIELDS = "paperId,title,year,authors,citationCount"
BATCH_SIZE = 500  # maximum number of IDs per /paper/batch request
MAX_RETRIES = 6

PAPER_SETS: dict[str, tuple[Path, Path]] = {
    "eval": (config.PAPER_IDS_EVAL_CSV, config.EVAL_SET_CSV),
    "proxy": (config.PAPER_IDS_PROXY_CSV, config.PROXY_SET_CSV),
    "proxy_resample": (config.PAPER_IDS_PROXY_RESAMPLE_CSV, config.PROXY_RESAMPLE_CSV),
}
EVAL_COLUMNS = ["paperId", "title", "year", "authors-info", "citationCount", "citation_bin", "field", "full_name_list"]
PROXY_COLUMNS = [
    "paperId", "title", "year", "authors-info", "citationCount", "full_name_list", "last_name_list",
    "last_name_list_w_quotes", "full_name_list_w_quotes", "questions", "citation_bin",
]


def fetch_batch(paper_ids: list[str], headers: dict[str, str]) -> list[dict | None]:
    """Fetch metadata for up to 500 papers, retrying on rate limits and server errors.

    Args:
        paper_ids: Semantic Scholar paper IDs.
        headers: HTTP headers (including ``x-api-key`` when available).

    Returns:
        One record per requested ID (None if Semantic Scholar does not know the ID).

    Raises:
        RuntimeError: If the request keeps failing after all retries.
    """
    delay = 2.0
    for _ in range(MAX_RETRIES):
        response = requests.post(
            S2_BATCH_URL, params={"fields": S2_FIELDS}, json={"ids": paper_ids}, headers=headers, timeout=60
        )
        if response.status_code == 200:
            return response.json()
        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue
        response.raise_for_status()
    raise RuntimeError("Semantic Scholar batch request failed after all retries")


def fetch_records(paper_ids: list[str], headers: dict[str, str]) -> dict[str, dict]:
    """Fetch all papers in batches.

    Args:
        paper_ids: Semantic Scholar paper IDs.
        headers: HTTP headers.

    Returns:
        A mapping from requested paper ID to its Semantic Scholar record (missing IDs are omitted).
    """
    records: dict[str, dict] = {}
    for start in tqdm(range(0, len(paper_ids), BATCH_SIZE), desc="Fetching from Semantic Scholar"):
        batch_ids = paper_ids[start:start + BATCH_SIZE]
        for pid, record in zip(batch_ids, fetch_batch(batch_ids, headers)):
            if record is not None:
                records[pid] = record
        time.sleep(1.1)
    return records


def quote_all(items: list[str]) -> str:
    """Wrap every item in double quotes and join them with spaces."""
    return " ".join(f'"{item}"' for item in items)


def build_rows(ids_df: pd.DataFrame, records: dict[str, dict], proxy: bool) -> pd.DataFrame:
    """Combine the stored snapshot values with the fetched titles and years (and authors if not stored).

    Args:
        ids_df: The released ID file (paperId, citationCount, citation_bin[, field, authors-info]).
        records: Fetched Semantic Scholar records keyed by paper ID.
        proxy: Build the proxy-set columns (search queries, last names) instead of the evaluation-set columns.

    Returns:
        The rebuilt paper set in the format of steps 5 / 6.
    """
    rows = []
    for snapshot in ids_df.to_dict("records"):
        record = records.get(snapshot["paperId"])
        if record is None:
            continue
        if "authors-info" in snapshot:  # evaluation set: use the stored snapshot authors
            authors_info = ast.literal_eval(snapshot["authors-info"])
        else:
            authors_info = [{"authorId": a.get("authorId"), "name": a.get("name")} for a in record.get("authors") or []]
        names = [a["name"] for a in authors_info]
        row = {**snapshot, "title": record.get("title"), "year": record.get("year"),
               "authors-info": authors_info, "full_name_list": names}
        if proxy:
            last_names = [name.split(" ")[-1] for name in names]
            row["last_name_list"] = last_names
            row["last_name_list_w_quotes"] = quote_all(last_names)
            row["full_name_list_w_quotes"] = quote_all(names)
            row["questions"] = f'"{row["title"]}" {row["full_name_list_w_quotes"]}'
        row["_current_citation_count"] = record.get("citationCount")
        rows.append(row)
    return pd.DataFrame(rows)


def report_drift(df: pd.DataFrame) -> None:
    """Print how many papers would fall into a different citation bin with today's counts."""
    current_bin = pd.cut(df["_current_citation_count"], bins=config.CITATION_BIN_EDGES,
                         labels=config.CITATION_BIN_LABELS).astype(str)
    n_changed = int((current_bin != df["citation_bin"].astype(str)).sum())
    print(f"  {n_changed} papers would be in a different citation bin today "
          "(the stored September 2025 bins are used).")


def rebuild(set_name: str, headers: dict[str, str]) -> None:
    """Rebuild one paper set and save it."""
    ids_path, out_path = PAPER_SETS[set_name]
    ids_df = pd.read_csv(ids_path)
    print(f"[{set_name}] {len(ids_df)} papers from {ids_path.name}")
    records = fetch_records(ids_df["paperId"].tolist(), headers)

    proxy = set_name != "eval"
    df = build_rows(ids_df, records, proxy=proxy)
    report_drift(df)
    df = df[PROXY_COLUMNS if proxy else EVAL_COLUMNS]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} papers to {out_path}")
    if len(df) < len(ids_df):
        print(f"  WARNING: {len(ids_df) - len(df)} IDs were not found on Semantic Scholar")


def main() -> None:
    """Parse arguments and rebuild the requested paper set(s)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=[*PAPER_SETS, "all"], default="all", dest="paper_set")
    args = parser.parse_args()

    headers = {"User-Agent": "dreaming-up-authors/1.0"}
    api_key = os.environ.get("S2_API_KEY", "").strip()
    if api_key:
        headers["x-api-key"] = api_key
    else:
        print("S2_API_KEY not set: using the shared unauthenticated rate limit.")

    for set_name in PAPER_SETS if args.paper_set == "all" else [args.paper_set]:
        rebuild(set_name, headers)


if __name__ == "__main__":
    main()
