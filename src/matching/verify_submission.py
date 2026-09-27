"""
Submission #1 final-output verification.

Checks:
1. matching_results.tsv exists
2. Exact output columns
3. Expected row count
4. Every Source1 ID appears exactly once
5. No duplicate IDs inside matched_entity_ids
6. Every predicted match exists in candidate_pairs.tsv
7. Empty matched_entity_ids are allowed
8. Does not modify inference output or candidate pairs
"""

from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_PATH = Path("data/matching/matching_results.tsv")
CANDIDATES_PATH = Path("data/matching/candidate_pairs.tsv")

EXPECTED_ROW_COUNT = 1_732_544

EXPECTED_COLUMNS = [
    "source1_entity_id",
    "matched_entity_ids",
]


# ============================================================
# HELPERS
# ============================================================

def fail(message):
    raise ValueError(f"QA FAILED: {message}")


def load_tsv(path):
    if not path.exists():
        fail(f"File does not exist: {path}")

    if not path.is_file():
        fail(f"Path is not a file: {path}")

    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )


# ============================================================
# MAIN QA
# ============================================================

def main():

    print("=" * 70)
    print("SUBMISSION #1 FINAL OUTPUT VERIFICATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Check result file
    # --------------------------------------------------------

    print("\n[1/7] Checking matching_results.tsv...")

    results = load_tsv(RESULTS_PATH)

    print(f"Result rows: {len(results):,}")

    # --------------------------------------------------------
    # 2. Check exact columns
    # --------------------------------------------------------

    print("\n[2/7] Checking columns...")

    actual_columns = results.columns.tolist()

    if actual_columns != EXPECTED_COLUMNS:
        fail(
            "Columns are incorrect.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Actual:   {actual_columns}"
        )

    print("PASS: Columns are exactly:")
    for column in actual_columns:
        print(f"  - {column}")

    # --------------------------------------------------------
    # 3. Check row count
    # --------------------------------------------------------

    print("\n[3/7] Checking row count...")

    if len(results) != EXPECTED_ROW_COUNT:
        fail(
            f"Expected {EXPECTED_ROW_COUNT:,} rows, "
            f"but found {len(results):,}"
        )

    print(
        f"PASS: Row count = {len(results):,}"
    )

    # --------------------------------------------------------
    # 4. Check Source1 uniqueness
    # --------------------------------------------------------

    print("\n[4/7] Checking Source1 ID uniqueness...")

    if results["source1_entity_id"].eq("").any():
        fail("Empty source1_entity_id found.")

    duplicate_source1 = results[
        results["source1_entity_id"].duplicated(keep=False)
    ]

    if not duplicate_source1.empty:
        print(
            f"Duplicate Source1 IDs: "
            f"{duplicate_source1['source1_entity_id'].nunique():,}"
        )

        print(
            duplicate_source1[
                ["source1_entity_id"]
            ].head(10).to_string(index=False)
        )

        fail(
            "Every Source1 ID must appear exactly once."
        )

    print(
        f"PASS: {results['source1_entity_id'].nunique():,} "
        "unique Source1 IDs."
    )

    # --------------------------------------------------------
    # 5. Check duplicate matched IDs
    # --------------------------------------------------------

    print("\n[5/7] Checking duplicate matched IDs...")

    duplicate_match_rows = []

    for _, row in results.iterrows():

        matched = row["matched_entity_ids"].strip()

        # Empty matches are valid.
        if not matched:
            continue

        ids = [
            x.strip()
            for x in matched.split(",")
            if x.strip()
        ]

        if len(ids) != len(set(ids)):
            duplicate_match_rows.append(
                row["source1_entity_id"]
            )

    if duplicate_match_rows:

        print(
            "Examples with duplicate matched IDs:"
        )

        for source1_id in duplicate_match_rows[:10]:
            print(f"  {source1_id}")

        fail(
            f"Found duplicate matched IDs in "
            f"{len(duplicate_match_rows):,} output rows."
        )

    print("PASS: No duplicate matched IDs.")

    # --------------------------------------------------------
    # 6. Load candidate pairs and verify subset
    # --------------------------------------------------------

    print("\n[6/7] Checking candidate-pair constraint...")

    candidates = load_tsv(CANDIDATES_PATH)

    required_candidate_columns = {
        "source1_entity_id",
        "candidate_entity_id",
    }

    missing = (
        required_candidate_columns
        - set(candidates.columns)
    )

    if missing:
        fail(
            "Candidate file is missing columns: "
            + ", ".join(sorted(missing))
        )

    candidate_pairs = set(
        zip(
            candidates["source1_entity_id"],
            candidates["candidate_entity_id"],
        )
    )

    print(
        f"Candidate pairs loaded: "
        f"{len(candidate_pairs):,}"
    )

    invalid_matches = []

    predicted_match_count = 0
    empty_match_count = 0

    for _, row in results.iterrows():

        source1_id = row["source1_entity_id"]
        matched = row["matched_entity_ids"].strip()

        if not matched:
            empty_match_count += 1
            continue

        ids = [
            x.strip()
            for x in matched.split(",")
            if x.strip()
        ]

        predicted_match_count += len(ids)

        for candidate_id in ids:

            if (
                source1_id,
                candidate_id,
            ) not in candidate_pairs:

                invalid_matches.append(
                    (
                        source1_id,
                        candidate_id,
                    )
                )

    if invalid_matches:

        print("\nExamples of invalid matches:")

        for source1_id, candidate_id in invalid_matches[:10]:
            print(
                f"  Source1={source1_id} "
                f"Candidate={candidate_id}"
            )

        fail(
            f"Found {len(invalid_matches):,} predicted matches "
            "that are NOT present in candidate_pairs.tsv."
        )

    print(
        "PASS: Every predicted match is present "
        "in candidate_pairs.tsv."
    )

    # --------------------------------------------------------
    # 7. Empty matches and summary
    # --------------------------------------------------------

    print("\n[7/7] Checking empty-match behavior...")

    print(
        f"Rows with empty matched_entity_ids: "
        f"{empty_match_count:,}"
    )

    print(
        f"Rows with at least one match: "
        f"{len(results) - empty_match_count:,}"
    )

    print(
        f"Total predicted candidate matches: "
        f"{predicted_match_count:,}"
    )

    print("\n" + "=" * 70)
    print("QA PASSED")
    print("=" * 70)

    print("\nVerified:")
    print("  PASS - Result file exists")
    print("  PASS - Exact output columns")
    print("  PASS - Expected row count")
    print("  PASS - Every Source1 ID appears exactly once")
    print("  PASS - No duplicate matched IDs")
    print("  PASS - All predicted IDs are candidate IDs")
    print("  PASS - Empty matches are allowed")
    print("  PASS - Candidate file was read only")
    print("=" * 70)


if __name__ == "__main__":
    main()