"""Step 4: keep the core Semantic Scholar columns and build author lists and search queries.

For every matched chunk this keeps paperId, title, year, authors and citation
count, and adds:
    full_name_list            list of author names
    last_name_list            list of author last names
    full_name_list_w_quotes   '"Name 1" "Name 2" ...'
    last_name_list_w_quotes   '"Last1" "Last2" ...'
    questions                 '"Paper Title" "Name 1" "Name 2" ...'  (Google query)

Input:
    config.MATCHED_CHUNKS_DIR / chunk_{i}.csv
Output:
    config.QUERY_FIELDS_DIR / samples_google_search_{i}.csv

Usage:
    python 01_data_collection/04_build_query_fields.py
"""

import ast
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

COLUMN_MAP = {
    "semantic_scholar.paperId": "paperId",
    "semantic_scholar.title": "title",
    "semantic_scholar.year": "year",
    "semantic_scholar.authors": "authors-info",
    "semantic_scholar.citationCount": "citationCount",
}


def author_names(authors_info: list[dict]) -> list[str]:
    """Extract author names from Semantic Scholar author records.

    Args:
        authors_info: List of ``{"authorId": ..., "name": ...}`` dictionaries.

    Returns:
        The author names in order.
    """
    return [author["name"] for author in authors_info]


def last_names(names: list[str]) -> list[str]:
    """Return the last whitespace-separated token of each name.

    Args:
        names: Full author names.

    Returns:
        The last names.
    """
    return [name.split(" ")[-1] for name in names]


def quote_all(items: list[str]) -> str:
    """Wrap every item in double quotes and join them with spaces.

    Args:
        items: Strings to quote.

    Returns:
        A string such as '"A" "B"'.
    """
    return " ".join(f'"{item}"' for item in items)


def build_query_fields(chunk: pd.DataFrame) -> pd.DataFrame:
    """Build author-list and query columns for one matched chunk.

    Args:
        chunk: A flattened matched chunk (all columns as strings).

    Returns:
        The reduced DataFrame with the derived columns.
    """
    df = chunk[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)
    df["authors-info"] = df["authors-info"].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else {})
    df["full_name_list"] = df["authors-info"].apply(author_names)
    df["last_name_list"] = df["full_name_list"].apply(last_names)
    df["last_name_list_w_quotes"] = df["last_name_list"].apply(quote_all)
    df["full_name_list_w_quotes"] = df["full_name_list"].apply(quote_all)
    df["questions"] = '"' + df["title"] + '" ' + df["full_name_list_w_quotes"]
    return df


def main() -> None:
    """Process every matched chunk."""
    config.QUERY_FIELDS_DIR.mkdir(parents=True, exist_ok=True)
    chunk_paths = sorted(config.MATCHED_CHUNKS_DIR.glob("chunk_*.csv"), key=lambda p: int(p.stem.split("_")[1]))
    for path in chunk_paths:
        i = int(path.stem.split("_")[1])
        chunk = pd.read_csv(path, dtype=str)
        out_path = config.QUERY_FIELDS_DIR / f"samples_google_search_{i}.csv"
        build_query_fields(chunk).to_csv(out_path, index=False)
        print(f"Saved {len(chunk)} papers to {out_path}")


if __name__ == "__main__":
    main()
