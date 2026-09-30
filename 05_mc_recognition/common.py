"""Shared paths and helpers for the multiple-choice (MC) recognition experiment.

Three distractor constructions ("variants") are supported:

    original            three random same-field author teams of size k (main result)
    same_field_swap     the true team with one author replaced by a random same-field author
    collaborator_swap   the true team with one non-first author replaced by a real collaborator
                        of the first author (from Semantic Scholar)
"""

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

VARIANTS: list[str] = ["original", "same_field_swap", "collaborator_swap"]
RANDOM_SEED = 42
FUZZY_THRESHOLD = 90
SEP = ", "
POSITIONS: list[str] = ["A", "B", "C", "D"]
BIN_ORDER: list[str] = config.CITATION_BIN_LABELS
COLLABORATORS_JSON: Path = config.MC_DIR / "collaborator_swap" / "collaborators.json"


@dataclass(frozen=True)
class VariantPaths:
    """All files belonging to one MC variant."""

    root: Path

    @property
    def options_csv(self) -> Path:
        """MC options (four choices and the correct position) per paper."""
        return self.root / "options.csv"

    @property
    def batch_jsonl(self) -> Path:
        """OpenAI Batch API request file."""
        return self.root / "batch_input.jsonl"

    @property
    def batch_jsonl_retry(self) -> Path:
        """Batch request file for papers missing from a partial batch."""
        return self.root / "batch_input_retry.jsonl"

    @property
    def batch_id_file(self) -> Path:
        """ID of the submitted batch (used to resume polling)."""
        return self.root / "batch_id.txt"

    @property
    def batch_id_retry_file(self) -> Path:
        """ID of the submitted retry batch."""
        return self.root / "batch_id_retry.txt"

    @property
    def responses_csv(self) -> Path:
        """Parsed model answers."""
        return self.root / "responses.csv"

    @property
    def staging_csv(self) -> Path:
        """Answers of a partial batch, waiting for the retry batch."""
        return self.root / "responses_staging.csv"

    @property
    def results_csv(self) -> Path:
        """Per-paper correctness and recall metrics."""
        return self.root / "results.csv"

    @property
    def bin_summary_csv(self) -> Path:
        """Metrics aggregated per citation bin (overall, per field, per paper type)."""
        return self.root / "bin_summary.csv"

    @property
    def figures_dir(self) -> Path:
        """Figures produced by the analysis notebook."""
        return self.root / "figures"


def variant_paths(variant: str) -> VariantPaths:
    """Return the file locations of one variant and create its folder.

    Args:
        variant: One of ``VARIANTS``.

    Returns:
        The variant's paths.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}', expected one of {VARIANTS}")
    paths = VariantPaths(config.MC_DIR / variant)
    paths.root.mkdir(parents=True, exist_ok=True)
    return paths


def parse_name_list(raw: object) -> list[str]:
    """Parse a stringified author-name list (``full_name_list`` column).

    Args:
        raw: The raw cell value.

    Returns:
        The non-empty, stripped author names.
    """
    try:
        names = ast.literal_eval(str(raw))
        return [str(n).strip() for n in names if str(n).strip()]
    except (ValueError, SyntaxError, TypeError):
        return []


def parse_authors_info(raw: object) -> list[dict]:
    """Parse a stringified ``authors-info`` list of ``{"authorId", "name"}`` dictionaries."""
    try:
        return ast.literal_eval(str(raw))
    except (ValueError, SyntaxError, TypeError):
        return []


def normalize_name(name: str) -> str:
    """Normalize a name to ``"lastname f"`` (last name + first initial) for fuzzy matching."""
    parts = name.strip().split()
    if not parts:
        return name.lower()
    last = parts[-1].lower()
    first_initial = parts[0][0].lower() if len(parts) > 1 else ""
    return f"{last} {first_initial}".strip()


def is_excluded(candidate: str, excluded_norms: list[str]) -> bool:
    """Return True if ``candidate`` fuzzy-matches any of the normalized excluded names."""
    candidate_norm = normalize_name(candidate)
    return any(fuzz.ratio(candidate_norm, excluded) >= FUZZY_THRESHOLD for excluded in excluded_norms)


def filter_pool(pool: list[str], excluded_norms: list[str]) -> list[str]:
    """Remove every name that fuzzy-matches one of the excluded (true) authors."""
    return [name for name in pool if not is_excluded(name, excluded_norms)]
