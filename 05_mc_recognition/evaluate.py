"""Score the multiple-choice answers of one variant and compare them with open-ended recall.

Per paper: MC correctness (chosen letter == correct position) and the recall
metrics HR1-HR5 plus binary perfect recall (H = U = 0), computed from the
GPT-4o self-evaluation counts of module 04.

Per citation bin (overall, per field, and per single-/multi-author papers):
    acc_mc                raw MC accuracy
    acc_mc_cc             chance-corrected accuracy (acc - 0.25) / 0.75, with a Wilson 95% CI
    acc_recall_hr{1..5}   recall accuracy 1 - HR
    acc_recall_perfect    share of papers with perfect recall
    gap_*                 recall-recognition gap = acc_mc_cc - recall accuracy
    residual_rate_50/75   share of papers with HR2 > 0.5 (0.75) that are still recognized
Chance correction is applied only after aggregation, never per paper.

Input:
    results/mc_recognition/<variant>/options.csv, responses.csv
    config.GPT4O_SELF_EVAL_CSV
Output:
    results/mc_recognition/<variant>/results.csv, bin_summary.csv

Usage:
    python 05_mc_recognition/evaluate.py --variant original [--force]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import BIN_ORDER, config  # noqa: E402


def _hr1(x: int, y: int, z: int) -> float:
    """Max-normalized error rate: 1 - M / max(M + H, M + U)."""
    denom = max(y + x, z + x)
    return 1 - x / denom if denom > 0 else 0.0


def _hr2(x: int, y: int, z: int) -> float:
    """Jaccard hallucination rate: (H + U) / (M + H + U)."""
    denom = y + x + z
    return (y + z) / denom if denom > 0 else 0.0


def _hr3(x: int, y: int, z: int) -> float:
    """Harmonic hallucination rate: 2HU / ((M + H)U + (M + U)H)."""
    if y == 0 and z == 0:
        return 0.0
    denom = (y + x) * z + (x + z) * y
    return (2 * y * z) / denom if denom > 0 else 0.0


def _hr4(x: int, y: int, z: int) -> float:
    """Hallucination-Only Rate: fabricated names as fraction of model output."""
    denom = x + y
    return y / denom if denom > 0 else 0.0


def _hr5(x: int, y: int, z: int) -> float:
    """Omission-Only Rate: missed true authors as fraction of ground-truth set."""
    denom = x + z
    return z / denom if denom > 0 else 0.0


def parse_eval(raw: object) -> tuple[int, int, int]:
    """Parse an "X, Y, Z" self-evaluation into (matched, hallucinated, omitted); failures give (0, 0, 0)."""
    try:
        parts = [p.strip() for p in str(raw).split(",")]
        return int(parts[0]), int(parts[1]), int(parts[2])
    except (ValueError, IndexError):
        return (0, 0, 0)


def chance_correct(acc: float) -> float:
    """Remove the 25% guessing baseline of a 4-option question."""
    return (acc - 0.25) / 0.75


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (center - half, center + half)


def aggregate_bin(group: pd.DataFrame, label: str) -> dict:
    """Aggregate the per-paper results of one stratum (see the module docstring for the metrics)."""
    n = len(group)
    n_correct = group["correct"].sum()
    acc = n_correct / n if n > 0 else float("nan")
    acc_cc = chance_correct(acc)

    ci_lo, ci_hi = wilson_ci(int(n_correct), n)
    cc_ci_lo = chance_correct(ci_lo)
    cc_ci_hi = chance_correct(ci_hi)

    acc_recall1 = 1 - group["hr1"].mean()
    acc_recall2 = 1 - group["hr2"].mean()
    acc_recall3 = 1 - group["hr3"].mean()
    acc_recall4 = 1 - group["hr4"].mean()
    acc_recall5 = 1 - group["hr5"].mean()
    acc_recall_perfect = group["perfect_recall"].mean() if "perfect_recall" in group.columns else float("nan")

    gap1 = acc_cc - acc_recall1
    gap2 = acc_cc - acc_recall2
    gap3 = acc_cc - acc_recall3
    gap4 = acc_cc - acc_recall4
    gap5 = acc_cc - acc_recall5
    gap_perfect = acc_cc - acc_recall_perfect

    recall_fail = group["hr2"] > 0.5
    n_recall_fail = recall_fail.sum()
    residual_rate = (
        group.loc[recall_fail, "correct"].sum() / n_recall_fail
        if n_recall_fail > 0
        else float("nan")
    )

    recall_fail_75 = group["hr2"] > 0.75
    n_recall_fail_75 = recall_fail_75.sum()
    residual_rate_75 = (
        group.loc[recall_fail_75, "correct"].sum() / n_recall_fail_75
        if n_recall_fail_75 > 0
        else float("nan")
    )

    return {
        "stratum": label,
        "n": n,
        "n_correct": int(n_correct),
        "acc_mc": acc,
        "acc_mc_cc": acc_cc,
        "acc_mc_cc_ci95_lo": cc_ci_lo,
        "acc_mc_cc_ci95_hi": cc_ci_hi,
        "acc_recall_hr1": acc_recall1,
        "acc_recall_hr2": acc_recall2,
        "acc_recall_hr3": acc_recall3,
        "acc_recall_hr4": acc_recall4,
        "acc_recall_hr5": acc_recall5,
        "acc_recall_perfect": acc_recall_perfect,
        "gap_hr1": gap1,
        "gap_hr2": gap2,
        "gap_hr3": gap3,
        "gap_hr4": gap4,
        "gap_hr5": gap5,
        "gap_perfect": gap_perfect,
        "n_recall_fail_50": int(n_recall_fail),
        "residual_rate_50": residual_rate,
        "n_recall_fail_75": int(n_recall_fail_75),
        "residual_rate_75": residual_rate_75,
    }


def build_bin_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per citation bin: overall, per field, and per paper type (single/multi author)."""
    rows = []

    # Overall (all fields, all paper types)
    for citation_bin in BIN_ORDER:
        group = results[results["citation_bin"] == citation_bin]
        if len(group) == 0:
            continue
        row = aggregate_bin(group, "overall")
        row["citation_bin"] = citation_bin
        row["field"] = "all"
        row["paper_type"] = "all"
        rows.append(row)

    # Per field
    for field in sorted(results["field"].unique()):
        for citation_bin in BIN_ORDER:
            group = results[
                (results["field"] == field) & (results["citation_bin"] == citation_bin)
            ]
            if len(group) == 0:
                continue
            row = aggregate_bin(group, field)
            row["citation_bin"] = citation_bin
            row["field"] = field
            row["paper_type"] = "all"
            rows.append(row)

    # Per paper type (single vs multi)
    for ptype in ["single", "multi"]:
        for citation_bin in BIN_ORDER:
            group = results[
                (results["paper_type"] == ptype) & (results["citation_bin"] == citation_bin)
            ]
            if len(group) == 0:
                continue
            row = aggregate_bin(group, ptype)
            row["citation_bin"] = citation_bin
            row["field"] = "all"
            row["paper_type"] = ptype
            rows.append(row)

    df = pd.DataFrame(rows)
    cols = ["citation_bin", "field", "paper_type", "stratum"] + [
        c for c in df.columns if c not in {"citation_bin", "field", "paper_type", "stratum"}
    ]
    return df[cols]


def main() -> None:
    """Evaluate one variant and save the per-paper results and the bin summary."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=common.VARIANTS, required=True)
    parser.add_argument("--force", action="store_true", help="overwrite existing output files")
    args = parser.parse_args()
    paths = common.variant_paths(args.variant)

    if not args.force and paths.results_csv.exists() and paths.bin_summary_csv.exists():
        print("Output files already exist; use --force to overwrite.")
        return
    if not paths.responses_csv.exists():
        sys.exit(f"{paths.responses_csv} not found: run query_gpt4o.py first.")

    mc_options = pd.read_csv(paths.options_csv)
    responses = pd.read_csv(paths.responses_csv)
    eval_df = pd.read_csv(config.GPT4O_SELF_EVAL_CSV)
    print(f"options: {len(mc_options)}, responses: {len(responses)}, recall evaluations: {len(eval_df)}")

    mc_options["paper_type"] = mc_options["n_authors"].apply(lambda k: "single" if k == 1 else "multi")
    df = mc_options.merge(responses[["paperId", "parsed_letter", "is_valid"]], on="paperId", how="left")
    n_missing = df["parsed_letter"].isna().sum()
    if n_missing:
        print(f"WARNING: {n_missing} papers have no response")
    n_invalid = (~df["is_valid"].fillna(False).astype(bool)).sum()
    print(f"invalid responses: {n_invalid} ({n_invalid / len(df):.4f})")
    df["correct"] = (df["parsed_letter"] == df["correct_pos"]).astype(int)

    # Per-paper recall metrics from the self-evaluation counts.
    eval_df[["_x", "_y", "_z"]] = eval_df["evaluation"].apply(lambda v: pd.Series(parse_eval(v)))
    for name, fn in [("hr1", _hr1), ("hr2", _hr2), ("hr3", _hr3), ("hr4", _hr4), ("hr5", _hr5)]:
        eval_df[name] = eval_df.apply(lambda r, fn=fn: fn(r["_x"], r["_y"], r["_z"]), axis=1)
    eval_df["perfect_recall"] = ((eval_df["_y"] == 0) & (eval_df["_z"] == 0)).astype(int)
    hr_cols = ["hr1", "hr2", "hr3", "hr4", "hr5", "perfect_recall"]
    df = df.merge(eval_df[["paperId"] + hr_cols], on="paperId", how="left")
    if df["hr1"].isna().any():
        print(f"WARNING: {df['hr1'].isna().sum()} papers have no recall evaluation")

    result_cols = ["paperId", "field", "citation_bin", "paper_type", "n_authors",
                   "correct", "parsed_letter", "correct_pos"] + hr_cols
    df[result_cols].to_csv(paths.results_csv, index=False)
    print(f"Saved {paths.results_csv} ({len(df)} rows)")

    bin_summary = build_bin_summary(df)
    bin_summary.to_csv(paths.bin_summary_csv, index=False)
    print(f"Saved {paths.bin_summary_csv} ({len(bin_summary)} rows)")

    overall = bin_summary[(bin_summary["field"] == "all") & (bin_summary["paper_type"] == "all")]
    print(f"\nOverall results by citation bin ({args.variant}):")
    print(overall[["citation_bin", "n", "acc_mc", "acc_mc_cc", "gap_hr2", "residual_rate_50"]].to_string(index=False))


if __name__ == "__main__":
    main()
