"""Ask GPT-4o the multiple-choice questions of one variant through the OpenAI Batch API.

GPT-4o (temperature 0, max 5 output tokens) receives a system prompt, two
demonstration items, and the question "Who are the authors of the paper titled
'<title>' which was published in <year>?" with the four options A-D. The first
character of the answer is parsed as the chosen letter.

The script is a resumable state machine; simply re-run it until it finishes:
    1. responses.csv exists            -> nothing to do.
    2. batch_id_retry.txt exists       -> poll the retry batch, merge with the staged answers, save.
    3. batch_id.txt exists             -> poll the batch; if some answers are missing, submit a retry batch.
    4. otherwise                       -> build batch_input.jsonl, upload it, and submit the batch.
If more than 1% of the answers are not a valid letter the run aborts.

Requires OPENAI_API_KEY (environment or ``.env``), except with ``--dry-run``,
which only writes the request file and prints sample prompts.

Input:
    config.EVAL_SET_CSV (title and year)
    results/mc_recognition/<variant>/options.csv
Output:
    results/mc_recognition/<variant>/batch_input.jsonl, batch_id.txt, responses.csv

Usage:
    python 05_mc_recognition/query_gpt4o.py --variant original --dry-run
    python 05_mc_recognition/query_gpt4o.py --variant original
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import VariantPaths, config  # noqa: E402

MODEL = "gpt-4o"
TEMPERATURE = 0
MAX_TOKENS = 5

SYSTEM_PROMPT = (
    "You are a research assistant with expertise in academic literature. "
    "Answer each multiple choice question by selecting the letter (A, B, C, or D) "
    "that correctly identifies the complete author list of the paper. "
    "Output only the single letter of your answer, with no additional explanation."
)

# Demonstration items. They are kept exactly as used in the paper so that the requests are
# reproduced verbatim: the ``original`` run used a three-author second example, while the two
# harder variants were run later with a seven-author second example (whose title lacks its
# closing quote).
_EXAMPLE_1 = """\
---

Example 1 (single-author paper):

Question: Who are the authors of the paper titled 'A Transition from a Microscopic to a Macroscopic Approach to Steady State Detonation' which was published in 1995?

A: Haibin Xiao
B: T. Palvannan
C: C. S. Coffey
D: Sriparna Chakrabarti

Answer: C

---

"""

_EXAMPLE_2_ORIGINAL = """\
Example 2 (multi-author paper):

Question: Who are the authors of the paper titled 'Cooperative spectrum sensing in cognitive radio networks with weighted decision fusion schemes' which was published in 2012?

A: H. Urkowitz, P. Stoica, A. Paulraj
B: Y. Chen, G. Yu, Z. Zhang
C: M. Gandetto, C. Regazzoni, G. Dandrea
D: R. Tandra, A. Sahai, S. Mishra

Answer: B

---

"""

_EXAMPLE_2_SWAP_VARIANTS = """\
Example 2 (multi-author paper):

Question: Who are the authors of the paper titled 'Sleep structure and quantity are determined by behavioral transition probability in Drosophila melanogaster which was published in 2018?

A: Xiaoquan Guo, V. Grinevich, C. Donald, D. Ringler, K. Oka, D. Cunningham, I. Taitzoglou
B: T. Wiggin, Patricia R. Goodwin, Nathan C. Donelson, Chang Liu, K. Trinh, Subrahata Sanyal, Leslie C. Griffith
C: K. Gould, D. Clayton, G. Brownlee, M. Kita, Daniel R. Marshak, S. Carroll, M. Rebagliati
D: N. Hill-Kapturczak, W. Abdallah, D. Valeyre, T. Schumacher, M. Noguchi, N. Prat, G. D. da Silva

Answer: B

---

"""

_INSTRUCTION = "Now answer the following:\n\n"

FEW_SHOT_BY_VARIANT: dict[str, str] = {
    "original": _EXAMPLE_1 + _EXAMPLE_2_ORIGINAL + _INSTRUCTION,
    "same_field_swap": _EXAMPLE_1 + _EXAMPLE_2_SWAP_VARIANTS + _INSTRUCTION,
    "collaborator_swap": _EXAMPLE_1 + _EXAMPLE_2_SWAP_VARIANTS + _INSTRUCTION,
}


def build_user_message(row: pd.Series, title: str, year: int, few_shot: str) -> str:
    """Build the user message: demonstrations followed by the question and the options of one item."""
    question = (
        f"Question: Who are the authors of the paper titled '{title}' "
        f"which was published in {year}?\n\n"
        f"A: {row['opt_A']}\n"
        f"B: {row['opt_B']}\n"
        f"C: {row['opt_C']}\n"
        f"D: {row['opt_D']}\n\n"
        "Answer:"
    )
    return few_shot + question


def build_batch_request(row: pd.Series, title: str, year: int, few_shot: str) -> dict:
    """Build one Batch API request (custom_id = paperId)."""
    return {
        "custom_id": str(row["paperId"]),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(row, title, year, few_shot)},
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        },
    }


def write_jsonl(
    mc_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    out_path: Path,
    few_shot: str,
) -> None:
    """Write one Batch API request per MC item to ``out_path``."""
    title_year: dict[str, tuple[str, int]] = {
        str(row["paperId"]): (str(row["title"]), int(row["year"]))
        for _, row in dataset_df.iterrows()
    }
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in mc_df.iterrows():
            pid = str(row["paperId"])
            title, year = title_year[pid]
            f.write(json.dumps(build_batch_request(row, title, year, few_shot), ensure_ascii=False) + "\n")
    line_count = sum(1 for _ in open(out_path, encoding="utf-8"))
    print(f"  Wrote {line_count} requests to {out_path}")
    assert line_count == len(mc_df), f"JSONL line count {line_count} != {len(mc_df)}"


def poll_batch(client: OpenAI, batch_id: str) -> tuple[str, str | None]:
    """Poll until batch completes; return (output_file_id, error_file_id)."""
    print(f"  Polling batch {batch_id}...")
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        counts = batch.request_counts
        print(
            f"  status={status}  "
            f"completed={counts.completed}  failed={counts.failed}  total={counts.total}"
        )
        if status == "completed":
            return batch.output_file_id, batch.error_file_id
        if status in ("failed", "expired", "cancelled"):
            if batch.errors and batch.errors.data:
                for err in batch.errors.data[:5]:
                    print(f"  ERROR line={err.line} code={err.code}: {err.message}")
            raise RuntimeError(f"Batch ended with status: {status}")
        # OpenAI sometimes gets stuck in "finalizing" even when all requests
        # are processed. If output_file_id is already populated, proceed.
        if (
            status == "finalizing"
            and counts.total > 0
            and counts.completed + counts.failed == counts.total
            and batch.output_file_id
        ):
            print("  All requests processed; proceeding despite finalizing status.")
            return batch.output_file_id, batch.error_file_id
        time.sleep(60)


def download_file(client: OpenAI, file_id: str) -> list[dict]:
    """Download a Batch API output or error file as a list of JSON records."""
    content = client.files.content(file_id)
    return [json.loads(line) for line in content.text.strip().splitlines() if line.strip()]


def show_error_summary(client: OpenAI, error_file_id: str) -> None:
    """Print the most frequent error codes of a batch error file."""
    try:
        lines = download_file(client, error_file_id)
        codes: Counter = Counter()
        for item in lines:
            try:
                code = item["response"]["body"]["error"]["code"]
            except (KeyError, TypeError):
                code = "unknown"
            codes[code] += 1
        print(f"  Error breakdown ({len(lines)} total):")
        for code, count in codes.most_common(10):
            print(f"    {code}: {count}")
        if lines:
            sample = lines[0]
            try:
                print(f"  Sample (custom_id={sample['custom_id']}): "
                      f"{sample['response']['body']['error']}")
            except (KeyError, TypeError):
                print(f"  Sample: {sample}")
    except Exception as exc:  # noqa: BLE001 - diagnostics only, never fatal
        print(f"  Could not download error file: {exc}")


def parse_responses(results: list[dict]) -> pd.DataFrame:
    """Parse batch outputs into paperId, raw_response, parsed_letter and is_valid."""
    rows = []
    for item in results:
        pid = item["custom_id"]
        try:
            raw = item["response"]["body"]["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raw = ""
        raw = raw.strip()
        letter = raw.upper()[:1] if raw else ""
        is_valid = letter in {"A", "B", "C", "D"}
        rows.append(
            {"paperId": pid, "raw_response": raw, "parsed_letter": letter, "is_valid": is_valid}
        )
    df = pd.DataFrame(rows)
    invalid_rate = (~df["is_valid"]).mean()
    print(f"  Parsed {len(df)} responses  invalid_rate={invalid_rate:.4f}")
    return df


def submit_batch(client: OpenAI, jsonl_path: Path, id_file: Path) -> str:
    """Upload a request file, create a 24h batch, and store its ID in ``id_file``."""
    print(f"Uploading {jsonl_path.name}...")
    with open(jsonl_path, "rb") as f:
        upload = client.files.create(file=f, purpose="batch")
    print(f"  file_id={upload.id}")
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    id_file.write_text(batch.id)
    print(f"  batch_id={batch.id} saved to {id_file}")
    return batch.id


def dry_run_check(mc_df: pd.DataFrame, dataset_df: pd.DataFrame, few_shot: str) -> None:
    """Print the prompt of the first item and of the item with the most authors."""
    title_year: dict[str, tuple[str, int]] = {
        str(row["paperId"]): (str(row["title"]), int(row["year"]))
        for _, row in dataset_df.iterrows()
    }
    sample = mc_df.iloc[0]
    pid = str(sample["paperId"])
    title, year = title_year[pid]
    print("\n--- Sample prompt (first paper) ---")
    print(f"paperId: {pid}  n_authors: {sample['n_authors']}")
    print(build_user_message(sample, title, year, few_shot))
    print("--- End sample prompt ---\n")

    high_k = mc_df.loc[mc_df["n_authors"].idxmax()]
    pid2 = str(high_k["paperId"])
    title2, year2 = title_year[pid2]
    msg2 = build_user_message(high_k, title2, year2, few_shot)
    print(f"--- Sample prompt (highest k={high_k['n_authors']}) ---")
    print(msg2[:800], "..." if len(msg2) > 800 else "")
    print("--- End sample prompt ---\n")


def main(variant: str, dry_run: bool = False) -> None:
    """Run the batch state machine for one variant.

    Args:
        variant: One of ``common.VARIANTS``.
        dry_run: Only build the request file and print sample prompts.
    """
    few_shot = FEW_SHOT_BY_VARIANT[variant]
    paths: VariantPaths = common.variant_paths(variant)
    if paths.responses_csv.exists():
        print(f"{paths.responses_csv} already exists: nothing to do.")
        return

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not dry_run:
        sys.exit("ERROR: OPENAI_API_KEY is not set (see .env.example), or use --dry-run.")

    mc_df = pd.read_csv(paths.options_csv)
    dataset_df = pd.read_csv(config.EVAL_SET_CSV)
    print(f"{len(mc_df)} MC items, {len(dataset_df)} papers")
    expected = len(mc_df)

    if dry_run:
        dry_run_check(mc_df, dataset_df, few_shot)
        if not paths.batch_jsonl.exists():
            write_jsonl(mc_df, dataset_df, paths.batch_jsonl, few_shot)
        else:
            print(f"{paths.batch_jsonl} already exists, reusing it")
        print("--dry-run: stopping before upload.")
        return

    client = OpenAI(api_key=api_key)

    # State 2: a retry batch is in flight.
    if paths.batch_id_retry_file.exists():
        retry_batch_id = paths.batch_id_retry_file.read_text().strip()
        print(f"Resuming retry batch {retry_batch_id}...")
        retry_out_id, retry_err_id = poll_batch(client, retry_batch_id)
        if retry_err_id:
            show_error_summary(client, retry_err_id)
        retry_df = parse_responses(download_file(client, retry_out_id))
        merged_df = pd.concat([pd.read_csv(paths.staging_csv), retry_df], ignore_index=True)
        merged_df = merged_df.drop_duplicates(subset="paperId")
        if len(merged_df) < expected:
            print(f"WARNING: still missing {expected - len(merged_df)} papers after the retry")
        merged_df.to_csv(paths.responses_csv, index=False)
        print(f"Saved {paths.responses_csv} ({len(merged_df)} rows)")
        paths.batch_id_retry_file.unlink()
        return

    if not paths.batch_jsonl.exists():
        write_jsonl(mc_df, dataset_df, paths.batch_jsonl, few_shot)
    else:
        print(f"{paths.batch_jsonl} already exists, reusing it")

    # State 3: the main batch is in flight (or state 4: submit it now).
    if paths.batch_id_file.exists():
        batch_id = paths.batch_id_file.read_text().strip()
        print(f"Resuming batch {batch_id}...")
    else:
        batch_id = submit_batch(client, paths.batch_jsonl, paths.batch_id_file)

    output_file_id, error_file_id = poll_batch(client, batch_id)
    if error_file_id:
        show_error_summary(client, error_file_id)
    responses_df = parse_responses(download_file(client, output_file_id))

    if len(responses_df) == expected:
        invalid_rate = (~responses_df["is_valid"]).mean()
        if invalid_rate > 0.01:
            raise RuntimeError(f"Invalid answer rate {invalid_rate:.4f} exceeds 1%: revise the prompt.")
        responses_df.to_csv(paths.responses_csv, index=False)
        print(f"Saved {paths.responses_csv} ({len(responses_df)} rows)")
        return

    # Partial results: stage them and submit a retry batch for the missing papers.
    print(f"Partial results: {len(responses_df)}/{expected} papers received.")
    responses_df.to_csv(paths.staging_csv, index=False)
    missing_df = mc_df[~mc_df["paperId"].astype(str).isin(set(responses_df["paperId"].astype(str)))]
    print(f"Submitting a retry batch for {len(missing_df)} papers...")
    write_jsonl(missing_df, dataset_df, paths.batch_jsonl_retry, few_shot)
    submit_batch(client, paths.batch_jsonl_retry, paths.batch_id_retry_file)
    print("Retry batch submitted. Re-run this script to poll and merge.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=common.VARIANTS, required=True)
    parser.add_argument("--dry-run", action="store_true", help="only build the request file")
    args = parser.parse_args()
    main(args.variant, dry_run=args.dry_run)
