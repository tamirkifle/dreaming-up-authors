"""Fetch the collaborator pool of every first author (needed by the ``collaborator_swap`` variant).

For every unique Semantic Scholar authorId of a first author in the evaluation
set, all of the author's papers are fetched (paginated, up to 20 x 1,000
papers) and their coauthors are merged into ``{collaboratorId: name}``. The
true authors of the target paper are excluded later, in ``build_options.py``.

The run is resumable: authors already stored in ``collaborators.json`` are
skipped, and progress is saved every 50 authors. With ``S2_API_KEY`` set the
requests are sent without delay; otherwise the shared unauthenticated endpoint
is used at about 1 request per second (roughly 3 hours for ~9,000 authors).

Semantic Scholar data changes over time, so a fresh run can produce slightly
different collaborator pools (and therefore distractors) than in the paper.

Input:
    config.EVAL_SET_CSV
Output:
    results/mc_recognition/collaborator_swap/collaborators.json

Usage:
    python 05_mc_recognition/fetch_collaborators.py
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import config, parse_authors_info  # noqa: E402

S2_URL = "https://api.semanticscholar.org/graph/v1/author/{aid}/papers"
PAGE_LIMIT = 1000
MAX_PAGES = 20
SAVE_EVERY = 50


def http_get_json(url: str, headers: dict[str, str], max_retries: int = 6) -> dict | None:
    """GET a JSON document, retrying with exponential backoff on rate limits and server errors.

    Args:
        url: Request URL.
        headers: HTTP headers.
        max_retries: Maximum number of attempts.

    Returns:
        The decoded JSON, an empty result for unknown authors (HTTP 404), or None on failure.
    """
    delay = 1.2
    for _ in range(max_retries):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return {"data": []}
            if exc.code not in (429, 500, 502, 503, 504):
                print(f"    HTTP error {exc.code}")
                return None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            pass
        time.sleep(delay)
        delay = min(delay * 2, 30)
    return None


def fetch_collaborators(author_id: str, headers: dict[str, str], rate_sleep: float) -> dict[str, str] | None:
    """Union the coauthors of all papers of one author.

    Args:
        author_id: Semantic Scholar authorId.
        headers: HTTP headers (with ``x-api-key`` when available).
        rate_sleep: Seconds to wait after every request.

    Returns:
        ``{collaboratorId: name}`` (including the author), or None if a request failed.
    """
    collaborators: dict[str, str] = {}
    offset = 0
    for _ in range(MAX_PAGES):
        url = f"{S2_URL.format(aid=author_id)}?fields=authors&limit={PAGE_LIMIT}&offset={offset}"
        data = http_get_json(url, headers)
        if rate_sleep:
            time.sleep(rate_sleep)
        if data is None:
            return None
        for paper in data.get("data", []):
            for author in paper.get("authors", []):
                if author.get("authorId") and author.get("name"):
                    collaborators.setdefault(author["authorId"], author["name"])
        if not data.get("next"):
            break
        offset = data["next"]
    return collaborators


def main() -> None:
    """Fetch and store the collaborator pools of all first authors."""
    api_key = os.environ.get("S2_API_KEY", "").strip()
    headers = {"User-Agent": "dreaming-up-authors/1.0"}
    if api_key:
        headers["x-api-key"] = api_key
    rate_sleep = 0.0 if api_key else 1.1
    print(f"S2_API_KEY: {'set' if api_key else 'not set (unauthenticated, ~1 request/s)'}")

    df = pd.read_csv(config.EVAL_SET_CSV)
    first_author_ids = (
        df["authors-info"].apply(parse_authors_info)
        .apply(lambda authors: authors[0]["authorId"] if authors else None)
        .dropna().astype(str).unique().tolist()
    )
    print(f"{len(first_author_ids)} unique first authors")

    out_path = common.COLLABORATORS_JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    collaborators: dict[str, dict[str, str]] = json.loads(out_path.read_text()) if out_path.exists() else {}
    todo = [aid for aid in first_author_ids if aid not in collaborators]
    print(f"{len(collaborators)} already fetched, {len(todo)} remaining")

    for i, author_id in enumerate(todo):
        pool = fetch_collaborators(author_id, headers, rate_sleep)
        collaborators[author_id] = pool if pool is not None else {}
        if (i + 1) % SAVE_EVERY == 0:
            out_path.write_text(json.dumps(collaborators))
            print(f"  {i + 1}/{len(todo)}")
    out_path.write_text(json.dumps(collaborators))

    sizes = sorted(len(pool) for pool in collaborators.values())
    n_enough = sum(size >= 3 for size in sizes)
    print(f"Saved {len(collaborators)} collaborator pools to {out_path}")
    print(f"  authors with >= 3 collaborators: {n_enough} ({n_enough / len(sizes):.1%})")
    print(f"  median pool size: {sizes[len(sizes) // 2]}")


if __name__ == "__main__":
    main()
