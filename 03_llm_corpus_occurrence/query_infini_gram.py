"""Count how often each paper (title + first author) occurs in an open LLM pretraining corpus.

Uses the public Infini-gram API (https://infini-gram.io). For every paper in the
prevalence-proxy set the query is ``<title> AND <first author last name>``, where
the title is formatted in one of two ways:

    capitalize   "Attention is all you need AND Vaswani"   (main analysis)
    camel        "Attention Is All You Need AND Vaswani"   (query-format sensitivity check)

Corpora (Infini-gram index names):
    olmo2       v4_olmo-2-0325-32b-instruct_llama   (OLMo2-32B-Instruct corpus)
    redpajama   v4_rpj_llama_s4                     (RedPajama)

Progress is checkpointed every 200 queries, so an interrupted run resumes where it stopped.

Input:
    config.PROXY_SET_CSV          (default)
    config.PROXY_RESAMPLE_CSV     (--resample)
Output:
    config.INFINI_GRAM_DIR / infini_gram_{corpus}_{query_format}[_resample].csv

Usage:
    python 03_llm_corpus_occurrence/query_infini_gram.py --corpus olmo2 --query-format capitalize
    python 03_llm_corpus_occurrence/query_infini_gram.py --corpus olmo2 --query-format capitalize --resample
"""

import argparse
import ast
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

API_URL = "https://api.infini-gram.io/"
INDEXES = {
    "olmo2": "v4_olmo-2-0325-32b-instruct_llama",
    "redpajama": "v4_rpj_llama_s4",
}
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; research-script/1.0)",
    "Origin": "https://infini-gram.io",
    "Referer": "https://infini-gram.io/",
}
CHECKPOINT_EVERY = 200
EMPTY_RESULT = {"count": None, "latency": None, "approx": None, "token_ids": None, "tokens": None}


def make_query(row: pd.Series, query_format: str) -> str:
    """Build the Infini-gram query for one paper.

    Args:
        row: A proxy-set row with ``title`` and ``last_name_list`` columns.
        query_format: ``"capitalize"`` (first letter upper case) or ``"camel"`` (every word capitalized).

    Returns:
        The query string ``<formatted title> AND <first author last name>``.
    """
    first_author_last_name = ast.literal_eval(row["last_name_list"])[0]
    title = row["title"].strip().lower()
    title = title.capitalize() if query_format == "capitalize" else title.title()
    return f"{title} AND {first_author_last_name}"


def count_query(query: str, index: str, max_retries: int = 5, sleep_time: float = 5.0) -> dict:
    """Send one count query to the Infini-gram API with exponential backoff.

    Args:
        query: The query string.
        index: Infini-gram index name.
        max_retries: Maximum number of attempts.
        sleep_time: Base waiting time in seconds.

    Returns:
        The count, latency, approximation flag and tokenization returned by the API
        (all None if every attempt failed).
    """
    payload = {"index": index, "query_type": "count", "query": query}
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, json=payload, timeout=15, headers=HEADERS)
            if response.status_code == 200:
                data = response.json()
                return {
                    "count": data.get("count"),
                    "latency": data.get("latency"),
                    "approx": data.get("approx"),
                    "token_ids": data.get("token_ids", []),
                    "tokens": data.get("tokens", []),
                }
            wait = sleep_time * (2 ** attempt) if response.status_code in (403, 429) else sleep_time
            print(f"HTTP {response.status_code} for query '{query}', retrying in {wait:.0f}s")
            time.sleep(wait)
        except requests.RequestException as exc:
            print(f"Attempt {attempt + 1} failed for query '{query}': {exc}")
            time.sleep(sleep_time)
    return dict(EMPTY_RESULT)


def main() -> None:
    """Query every paper in the proxy set and save the counts."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", choices=sorted(INDEXES), required=True)
    parser.add_argument("--query-format", choices=["capitalize", "camel"], required=True)
    parser.add_argument("--resample", action="store_true", help="use the resampled proxy set")
    args = parser.parse_args()

    input_csv = config.PROXY_RESAMPLE_CSV if args.resample else config.PROXY_SET_CSV
    suffix = "_resample" if args.resample else ""
    out_path = config.INFINI_GRAM_DIR / f"infini_gram_{args.corpus}_{args.query_format}{suffix}.csv"
    partial_path = out_path.with_suffix(".partial.csv")
    config.INFINI_GRAM_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    df["query"] = df.apply(make_query, axis=1, query_format=args.query_format)

    results: list[dict] = []
    if partial_path.exists():
        results = pd.read_csv(partial_path).to_dict("records")
        print(f"Resuming after {len(results)} finished queries")

    index = INDEXES[args.corpus]
    for i in tqdm(range(len(results), len(df)), desc=f"Infini-gram {args.corpus}/{args.query_format}"):
        results.append(count_query(df["query"].iloc[i], index))
        time.sleep(1)
        if (i + 1) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(results).to_csv(partial_path, index=False)

    merged = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    merged.to_csv(out_path, index=False)
    partial_path.unlink(missing_ok=True)
    print(f"Saved {len(merged)} rows to {out_path} ({merged['count'].isna().sum()} failed queries)")


if __name__ == "__main__":
    main()
