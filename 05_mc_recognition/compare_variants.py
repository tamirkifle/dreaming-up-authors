"""Print the MC recognition tables and test statistics reported in the paper.

For every variant that has been evaluated, this prints
    * recall accuracy (1 - HR2), chance-corrected MC accuracy and the recall-recognition
      gap per citation bin (the full-results table and the harder-distractor table),
    * the residual recognition rate among papers whose recall fails (HR2 > 0.5 and > 0.75),
    * a one-sample t-test of MC accuracy against the 0.25 chance level in the 0-2 bin,
    * the Spearman correlation between citation bin and chance-corrected accuracy.
The combined table is also saved as a CSV.

Input:
    results/mc_recognition/<variant>/results.csv, bin_summary.csv
Output:
    results/mc_recognition/variant_comparison.csv

Usage:
    python 05_mc_recognition/compare_variants.py
"""

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import BIN_ORDER, config  # noqa: E402

SHORT_NAMES = {"original": "orig", "same_field_swap": "same_field", "collaborator_swap": "collab"}


def overall_by_bin(bin_summary: pd.DataFrame) -> pd.DataFrame:
    """Select the all-fields, all-paper-types rows, ordered by citation bin."""
    overall = bin_summary[(bin_summary["field"] == "all") & (bin_summary["paper_type"] == "all")]
    return overall.set_index("citation_bin").reindex(BIN_ORDER)


def report_variant(variant: str, results: pd.DataFrame, overall: pd.DataFrame) -> None:
    """Print the residual recognition rates and test statistics of one variant."""
    print(f"\n=== {variant} ===")
    for threshold in (0.5, 0.75):
        failed = results[results["hr2"] > threshold]
        print(f"HR2 > {threshold}: {len(failed)} papers, {int(failed['correct'].sum())} recognized "
              f"({failed['correct'].mean():.1%})")

    low = results[results["citation_bin"] == "0-2"]
    t_stat, p_value = stats.ttest_1samp(low["correct"], 0.25)
    low_failed = low[low["hr2"] > 0.5]
    print(f"0-2 bin: MC accuracy (chance-corrected) = {overall.loc['0-2', 'acc_mc_cc']:.3f}, "
          f"one-sample t = {t_stat:.1f} (p = {p_value:.2g}); "
          f"recognized among recall failures = {low_failed['correct'].mean():.1%}")

    rho, p_rho = stats.spearmanr(range(len(BIN_ORDER)), overall["acc_mc_cc"])
    print(f"Spearman rho(citation bin, MC accuracy) = {rho:.3f} (p = {p_rho:.2g})")

    pooled_cc = (results["correct"].mean() - 0.25) / 0.75
    pooled_gap = pooled_cc - (1 - results["hr2"].mean())
    print(f"Gap (HR2): pooled over all papers = {pooled_gap:.3f}; across the 13 bins: "
          f"median = {overall['gap_hr2'].median():.3f}, mean = {overall['gap_hr2'].mean():.3f}, "
          f"max = {overall['gap_hr2'].max():.3f}")


def main() -> None:
    """Load every evaluated variant, print the statistics, and save the comparison table."""
    table = pd.DataFrame(index=BIN_ORDER)
    for variant in common.VARIANTS:
        paths = common.variant_paths(variant)
        if not paths.bin_summary_csv.exists():
            print(f"(skipping {variant}: {paths.bin_summary_csv.name} not found)")
            continue
        overall = overall_by_bin(pd.read_csv(paths.bin_summary_csv))
        results = pd.read_csv(paths.results_csv)
        if "recall_hr2" not in table:
            table["n"] = overall["n"]
            table["recall_hr2"] = overall["acc_recall_hr2"]
            table["recall_perfect"] = overall["acc_recall_perfect"]
        name = SHORT_NAMES[variant]
        table[f"mc_acc_cc_{name}"] = overall["acc_mc_cc"]
        table[f"gap_{name}"] = overall["gap_hr2"]
        table[f"residual_rate_50_{name}"] = overall["residual_rate_50"]
        report_variant(variant, results, overall)

    if table.empty:
        sys.exit("No evaluated variant found: run evaluate.py first.")
    table.index.name = "citation_bin"
    print("\nRecall vs. recognition by citation bin:")
    print(table.round(3).to_string())
    out_path = config.MC_DIR / "variant_comparison.csv"
    table.to_csv(out_path)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
