"""Build the four-option multiple-choice items for every paper in the evaluation set.

Every item shows the complete true author list (k names) and three distractor
lists of exactly k names. The position of the correct option (A-D) is drawn
uniformly at random.

Variants:
    original
        Per-field author pool = all unique author names of that field in the
        evaluation set. 3k names that do not fuzzy-match a true author are
        sampled without replacement and split into three distractor teams. If
        the pool is too small, each team is sampled independently
        (``distractor_fallback`` = True).
    same_field_swap
        Reuses the correct option and its position from ``original``. Each
        distractor is the true, ordered author list with ONE position replaced
        by a random same-field author (so it shares k-1 true authors). The
        (position, name) pairs are distinct, so the three distractors differ.
    collaborator_swap
        Same as ``same_field_swap``, but the first author is kept fixed (k > 1)
        and the replacement name is a real collaborator of the first author,
        taken from ``collaborators.json`` (see ``fetch_collaborators.py``). If
        fewer than 3 usable collaborators exist, the same-field pool is used
        instead (``collab_fallback`` = True).

All variants use ``random.Random(42)``. ``same_field_swap`` and
``collaborator_swap`` require the ``original`` options to exist. If the options
file already exists, only the quality checks are run.

Input:
    config.EVAL_SET_CSV
    results/mc_recognition/original/options.csv            (swap variants)
    results/mc_recognition/collaborator_swap/collaborators.json
Output:
    results/mc_recognition/<variant>/options.csv

Usage:
    python 05_mc_recognition/build_options.py --variant original
    python 05_mc_recognition/build_options.py --variant same_field_swap
    python 05_mc_recognition/build_options.py --variant collaborator_swap
"""

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import SEP, config, filter_pool, is_excluded, normalize_name, parse_authors_info, parse_name_list  # noqa: E402

MIN_COLLABORATORS = 3
MAX_TRIES = {"same_field_swap": 2000, "collaborator_swap": 3000}


# ---------------------------------------------------------------------------
# Distractor construction
# ---------------------------------------------------------------------------
def build_field_pools(df: pd.DataFrame) -> dict[str, list[str]]:
    """Collect the unique author names of every field (exact-string deduplication, sorted).

    Args:
        df: The evaluation set.

    Returns:
        A mapping from field to its sorted author pool.
    """
    pools: dict[str, list[str]] = {}
    for field, group in df.groupby("field"):
        names: set[str] = set()
        for raw in group["full_name_list"]:
            names.update(parse_name_list(raw))
        pools[field] = sorted(names)
    return pools


def random_team_distractors(
    pool: list[str], excluded_norms: list[str], k: int, rng: random.Random
) -> tuple[list[str], bool]:
    """Sample three distractor teams of k names from a same-field pool.

    Args:
        pool: Same-field author pool.
        excluded_norms: Normalized true-author names.
        k: Number of true authors.
        rng: Seeded random number generator.

    Returns:
        The three comma-joined teams and whether the independent-sampling fallback was used.

    Raises:
        ValueError: If the pool holds fewer than k usable names.
    """
    candidates = filter_pool(pool, excluded_norms)
    if len(candidates) >= 3 * k:
        sampled = rng.sample(candidates, 3 * k)
        return [SEP.join(sampled[i * k:(i + 1) * k]) for i in range(3)], False
    if len(candidates) < k:
        raise ValueError(f"pool too small even for fallback: {len(candidates)} < k={k}")
    return [SEP.join(rng.sample(candidates, k)) for _ in range(3)], True


def one_swap_distractors(
    authors: list[str], source_names: list[str], swappable: list[int], rng: random.Random, max_tries: int
) -> list[str]:
    """Build three distinct distractors, each replacing one author position with a source name.

    Args:
        authors: Ordered true author list.
        source_names: Candidate replacement names (none of them is a true author).
        swappable: Positions that may be replaced.
        rng: Seeded random number generator.
        max_tries: Safety cap on rejection-sampling iterations.

    Returns:
        Three comma-joined distractor lists.

    Raises:
        ValueError: If three distinct distractors cannot be built.
    """
    if not source_names or len(swappable) * len(source_names) < 3:
        raise ValueError("not enough (position, replacement) combinations for 3 distractors")
    seen: set[tuple[int, str]] = set()
    distractors: list[str] = []
    for _ in range(max_tries):
        pos = rng.choice(swappable)
        replacement = rng.choice(source_names)
        if (pos, replacement) in seen:
            continue
        seen.add((pos, replacement))
        variant = authors.copy()
        variant[pos] = replacement
        distractors.append(SEP.join(variant))
        if len(distractors) == 3:
            return distractors
    raise ValueError("could not build 3 distinct distractors within max_tries")


def usable_collaborators(pool: dict[str, str], true_ids: set[str], excluded_norms: list[str]) -> list[str]:
    """Collaborator names that may replace a true author (not on the paper, no fuzzy match).

    Args:
        pool: The first author's collaborators as ``{authorId: name}``.
        true_ids: Semantic Scholar IDs of the paper's authors.
        excluded_norms: Normalized true-author names.

    Returns:
        The sorted, deduplicated usable names.
    """
    names: set[str] = set()
    for collaborator_id, name in pool.items():
        if collaborator_id in true_ids or not name or name in names:
            continue
        if not is_excluded(name, excluded_norms):
            names.add(name)
    return sorted(names)


def place_correct(distractors: list[str], correct_option: str, correct_pos: str) -> list[str]:
    """Insert the correct option at its letter position among the three distractors."""
    options = distractors.copy()
    options.insert(common.POSITIONS.index(correct_pos), correct_option)
    return options


def item_row(paper: pd.Series, k: int, correct_option: str, options: list[str], correct_pos: str) -> dict:
    """Assemble one output row."""
    return {
        "paperId": paper["paperId"],
        "field": paper["field"],
        "citation_bin": paper["citation_bin"],
        "n_authors": k,
        "correct_option": correct_option,
        "opt_A": options[0],
        "opt_B": options[1],
        "opt_C": options[2],
        "opt_D": options[3],
        "correct_pos": correct_pos,
    }


# ---------------------------------------------------------------------------
# Variant builders
# ---------------------------------------------------------------------------
def build_original(df: pd.DataFrame, pools: dict[str, list[str]], rng: random.Random) -> list[dict]:
    """Build the ``original`` items (random same-field teams)."""
    rows = []
    for i, (_, paper) in enumerate(df.iterrows()):
        if i % 1000 == 0:
            print(f"  {i}/{len(df)}")
        authors = parse_name_list(paper["full_name_list"])
        if not authors:
            raise ValueError(f"paper {paper['paperId']} has no authors")
        excluded_norms = [normalize_name(a) for a in authors]
        distractors, fallback = random_team_distractors(pools[paper["field"]], excluded_norms, len(authors), rng)
        correct_option = SEP.join(authors)
        correct_pos = rng.choice(common.POSITIONS)
        row = item_row(paper, len(authors), correct_option, place_correct(distractors, correct_option, correct_pos),
                       correct_pos)
        row["distractor_fallback"] = fallback
        rows.append(row)
    return rows


def build_swap(
    variant: str, df: pd.DataFrame, src_df: pd.DataFrame, pools: dict[str, list[str]], rng: random.Random
) -> list[dict]:
    """Build the ``same_field_swap`` or ``collaborator_swap`` items.

    The correct option and its position are copied from the ``original`` items.
    """
    collaborators: dict[str, dict[str, str]] = {}
    if variant == "collaborator_swap":
        if not common.COLLABORATORS_JSON.exists():
            sys.exit(f"{common.COLLABORATORS_JSON} not found: run fetch_collaborators.py first.")
        collaborators = json.loads(common.COLLABORATORS_JSON.read_text())
        print(f"  {len(collaborators)} collaborator pools loaded")

    authors_map, ids_map, first_author_map = {}, {}, {}
    for _, paper in df.iterrows():
        pid = str(paper["paperId"])
        authors_map[pid] = parse_name_list(paper["full_name_list"])
        info = parse_authors_info(paper["authors-info"])
        ids_map[pid] = {a["authorId"] for a in info if a.get("authorId")}
        first_author_map[pid] = info[0]["authorId"] if info and info[0].get("authorId") else None

    rows = []
    for i, (_, src) in enumerate(src_df.iterrows()):
        if i % 1000 == 0:
            print(f"  {i}/{len(src_df)}")
        pid = str(src["paperId"])
        authors = authors_map.get(pid, [])
        correct_option = str(src["correct_option"])
        if not authors or SEP.join(authors) != correct_option:
            raise ValueError(f"paper {pid}: evaluation-set authors do not match the original correct option")
        k = len(authors)
        excluded_norms = [normalize_name(a) for a in authors]

        if variant == "same_field_swap":
            source_names = filter_pool(pools[src["field"]], excluded_norms)
            swappable = list(range(k))
            fallback = False
        else:
            pool = collaborators.get(first_author_map.get(pid) or "", {})
            source_names = usable_collaborators(pool, ids_map.get(pid, set()), excluded_norms)
            fallback = len(source_names) < MIN_COLLABORATORS
            if fallback:
                source_names = filter_pool(pools[src["field"]], excluded_norms)
            # The first author stays fixed for multi-author papers.
            swappable = list(range(1, k)) if k > 1 else [0]

        distractors = one_swap_distractors(authors, source_names, swappable, rng, MAX_TRIES[variant])
        correct_pos = str(src["correct_pos"])
        row = item_row(src, k, correct_option, place_correct(distractors, correct_option, correct_pos), correct_pos)
        row["distractor_fallback" if variant == "same_field_swap" else "collab_fallback"] = fallback
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------
def distractors_of(row: pd.Series) -> list[str]:
    """The three options of an item that are not the correct option."""
    return [str(row[f"opt_{p}"]) for p in common.POSITIONS if str(row[f"opt_{p}"]) != str(row["correct_option"])]


def single_swap_position(authors: list[str], option: str, correct_norms: list[str]) -> int | None:
    """Return the swapped position if ``option`` equals ``authors`` with one non-author replacement.

    The check reconstructs the joined string for every position instead of
    splitting on commas, because some names contain commas (e.g. "Wagner Meira, Jr").
    """
    k = len(authors)
    for i in range(k):
        left = SEP.join(authors[:i]) + SEP if i > 0 else ""
        right = SEP + SEP.join(authors[i + 1:]) if i < k - 1 else ""
        if not (option.startswith(left) and option.endswith(right)):
            continue
        middle = option[len(left):len(option) - len(right)] if right else option[len(left):]
        if SEP.join(authors[:i] + [middle] + authors[i + 1:]) == option and not is_excluded(middle, correct_norms):
            return i
    return None


def quality_check(variant: str, mc_df: pd.DataFrame, df: pd.DataFrame, src_df: pd.DataFrame | None) -> None:
    """Assert the structural properties of the generated items.

    Raises:
        AssertionError: If any check fails.
    """
    pos_dist = mc_df["correct_pos"].value_counts(normalize=True)
    assert all(0.20 <= pos_dist.get(p, 0.0) <= 0.30 for p in common.POSITIONS), f"position bias: {pos_dist.to_dict()}"
    print("  [OK] correct_pos distribution:", {p: round(v, 3) for p, v in pos_dist.items()})
    assert len(mc_df) == len(df), f"row count {len(mc_df)} != {len(df)}"
    print(f"  [OK] row count = {len(mc_df)}")

    authors_map = {str(r["paperId"]): parse_name_list(r["full_name_list"]) for _, r in df.iterrows()}
    problems = 0
    for _, row in mc_df.iterrows():
        authors = authors_map[str(row["paperId"])]
        norms = [normalize_name(a) for a in authors]
        distractors = distractors_of(row)
        if len(distractors) != 3 or len(set(distractors)) != 3:
            problems += 1
            continue
        for option in distractors:
            if variant == "original":
                names = [n.strip() for n in option.split(",")]
                if any(is_excluded(n, norms) for n in names) or len(names) != len(set(names)):
                    problems += 1
            else:
                pos = single_swap_position(authors, option, norms)
                if pos is None or (variant == "collaborator_swap" and len(authors) > 1 and pos == 0):
                    problems += 1
    assert problems == 0, f"{problems} invalid distractors"
    print("  [OK] every item has 3 distinct, valid distractors")

    if src_df is not None:
        src = src_df.set_index(src_df["paperId"].astype(str))
        changed = sum(
            str(row["correct_option"]) != str(src.at[str(row["paperId"]), "correct_option"])
            or str(row["correct_pos"]) != str(src.at[str(row["paperId"]), "correct_pos"])
            for _, row in mc_df.iterrows()
        )
        assert changed == 0, f"{changed} items changed the correct option or its position"
        print("  [OK] correct option and position identical to the original variant")

    fallback_col = "collab_fallback" if variant == "collaborator_swap" else "distractor_fallback"
    n_fallback = int(mc_df[fallback_col].sum())
    print(f"  [INFO] {fallback_col}: {n_fallback} items ({n_fallback / len(mc_df):.1%})")


def main() -> None:
    """Build (or verify) the options of one variant."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=common.VARIANTS, required=True)
    args = parser.parse_args()
    paths = common.variant_paths(args.variant)

    df = pd.read_csv(config.EVAL_SET_CSV)
    pools = build_field_pools(df)
    print(f"{len(df)} papers; author pool sizes: { {f: len(p) for f, p in sorted(pools.items())} }")

    src_df = None
    if args.variant != "original":
        original_csv = common.variant_paths("original").options_csv
        if not original_csv.exists():
            sys.exit(f"{original_csv} not found: build the 'original' variant first.")
        src_df = pd.read_csv(original_csv)

    if paths.options_csv.exists():
        print(f"{paths.options_csv} already exists; running quality checks only.")
        quality_check(args.variant, pd.read_csv(paths.options_csv), df, src_df)
        return

    print(f"Building '{args.variant}' options...")
    rng = random.Random(common.RANDOM_SEED)
    if args.variant == "original":
        rows = build_original(df, pools, rng)
    else:
        rows = build_swap(args.variant, df, src_df, pools, rng)
    mc_df = pd.DataFrame(rows)

    print("Running quality checks...")
    quality_check(args.variant, mc_df, df, src_df)
    mc_df.to_csv(paths.options_csv, index=False)
    print(f"Saved {len(mc_df)} items to {paths.options_csv}")


if __name__ == "__main__":
    main()
