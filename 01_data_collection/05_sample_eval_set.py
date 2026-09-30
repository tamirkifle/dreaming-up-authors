"""Step 5: build the stratified evaluation set (8 fields x 13 citation bins, up to 100 per stratum).

The matched chunks are scanned in file order (chunk_0, chunk_1, ...) and each
paper is assigned to a citation bin (Semantic Scholar citation count) and to the
first of its Semantic Scholar ``fieldsOfStudy`` that is one of the eight target
fields. A paper is kept while its (field, bin) stratum holds fewer than 100
papers. Because the input chunks are themselves a random sample of OAG, this
greedy fill yields a random sample within each stratum.

The target is 100 x 8 x 13 = 10,400 papers; high-citation strata cannot be
filled, and papers without a publication year are dropped, giving 9,108 papers.

The exact 9,108 Semantic Scholar paper IDs used in the paper are listed in
``01_data_collection/paper_ids_9108.csv``; ``07_fetch_datasets_from_ids.py``
rebuilds this file from those IDs without running steps 1-4.

Input:
    config.MATCHED_CHUNKS_DIR / chunk_{i}.csv
Output:
    config.EVAL_SET_RAW_CSV   all OAG + Semantic Scholar columns
    config.EVAL_SET_CSV       paperId, title, year, authors-info, citationCount,
                              citation_bin, field, full_name_list

Usage:
    python 01_data_collection/05_sample_eval_set.py
"""

import ast
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

MAX_PER_STRATUM = 100
CITATION_COL = "semantic_scholar.citationCount"
EXTRACT_COLUMNS = {
    "semantic_scholar.paperId": "paperId",
    "semantic_scholar.title": "title",
    "semantic_scholar.year": "year",
    "semantic_scholar.authors": "authors-info",
    "semantic_scholar.citationCount": "citationCount",
    "citation_bin": "citation_bin",
    "field": "field",
}


def assign_field(fields_of_study: object) -> str | None:
    """Return the first target field listed in a paper's Semantic Scholar fieldsOfStudy.

    Args:
        fields_of_study: The raw ``fieldsOfStudy`` cell (a stringified list).

    Returns:
        The lower-cased field name, or None if no target field is listed.
    """
    if not isinstance(fields_of_study, str) or not fields_of_study:
        return None
    fields_list = ast.literal_eval(fields_of_study)
    if not isinstance(fields_list, list):
        return None
    for field in fields_list:
        if field.lower() in config.FIELDS:
            return field.lower()
    return None


def sample_strata() -> pd.DataFrame:
    """Scan the matched chunks and fill every (bin, field) stratum greedily.

    Returns:
        The selected rows (all columns), before dropping papers without a year.
    """
    labels = config.CITATION_BIN_LABELS
    max_per_bin = MAX_PER_STRATUM * len(config.FIELDS)
    target_total = max_per_bin * len(labels)

    bin_rows: dict[str, list[pd.Series]] = {label: [] for label in labels}
    bin_counts = {label: 0 for label in labels}
    stratum_counts = {label: {field: 0 for field in config.FIELDS} for label in labels}

    chunk_paths = sorted(config.MATCHED_CHUNKS_DIR.glob("chunk_*.csv"), key=lambda p: int(p.stem.split("_")[1]))
    for path in tqdm(chunk_paths, desc="Scanning chunks"):
        columns = pd.read_csv(path, nrows=0).columns
        dtypes = {col: str for col in columns if col != CITATION_COL}
        data = pd.read_csv(path, dtype=dtypes, low_memory=False)
        data["citation_bin"] = pd.cut(data[CITATION_COL], bins=config.CITATION_BIN_EDGES, labels=labels)

        for _, row in data.iterrows():
            # Note: a missing year is NaN (truthy) here; such rows are dropped after sampling.
            if not row["semantic_scholar.year"]:
                continue
            bin_label = row["citation_bin"]
            if bin_counts[bin_label] >= max_per_bin:
                continue
            field = assign_field(row["semantic_scholar.fieldsOfStudy"])
            if field is None or stratum_counts[bin_label][field] >= MAX_PER_STRATUM:
                continue

            row["field"] = field
            bin_rows[bin_label].append(row)
            bin_counts[bin_label] += 1
            stratum_counts[bin_label][field] += 1
            if sum(bin_counts.values()) >= target_total:
                break
        if sum(bin_counts.values()) >= target_total:
            break

    return pd.concat([pd.DataFrame(bin_rows[label]) for label in labels], ignore_index=True)


def extract_columns(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Reduce the sampled rows to the columns used by the experiments.

    Args:
        raw_df: Sampled rows with all columns.

    Returns:
        The compact evaluation set with an added ``full_name_list`` column.
    """
    df = raw_df[list(EXTRACT_COLUMNS)].rename(columns=EXTRACT_COLUMNS)
    df["authors-info"] = df["authors-info"].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else {})
    df["full_name_list"] = df["authors-info"].apply(lambda authors: [a["name"] for a in authors])
    return df


def main() -> None:
    """Sample the evaluation set and write both output files."""
    raw_df = sample_strata()
    print(f"Selected {len(raw_df)} papers before dropping missing years")
    raw_df = raw_df.dropna(subset=["semantic_scholar.year"])
    print(f"Final evaluation set: {len(raw_df)} papers")
    print(raw_df["citation_bin"].value_counts(sort=False).to_string())
    print(raw_df["field"].value_counts().to_string())

    config.EVAL_SET_DIR.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(config.EVAL_SET_RAW_CSV, index=False)
    extract_columns(raw_df).to_csv(config.EVAL_SET_CSV, index=False)
    print(f"Saved {config.EVAL_SET_RAW_CSV}\nSaved {config.EVAL_SET_CSV}")


if __name__ == "__main__":
    main()
