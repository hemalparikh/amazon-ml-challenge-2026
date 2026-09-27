import argparse
import csv
import os
import sys


REQUIRED_COLUMNS = {
    "source1_entity_id",
    "candidate_entity_ids",
}


def load_entity_ids(path):
    """Load entity_id values from a TSV file."""
    ids = set()

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        if "entity_id" not in (reader.fieldnames or []):
            raise ValueError(
                f"{path} is missing required column: entity_id"
            )

        for row in reader:
            entity_id = (row.get("entity_id") or "").strip()

            if entity_id:
                ids.add(entity_id)

    return ids


def validate(candidate_path, test_dir):
    errors = []

    source1_path = os.path.join(test_dir, "test_source1.tsv")
    source2_path = os.path.join(test_dir, "test_source2.tsv")
    source3_path = os.path.join(test_dir, "test_source3.tsv")

    print("Loading Source1 IDs...")
    source1_ids = load_entity_ids(source1_path)

    print("Loading Source2 IDs...")
    source2_ids = load_entity_ids(source2_path)

    print("Loading Source3 IDs...")
    source3_ids = load_entity_ids(source3_path)

    valid_candidate_ids = source2_ids | source3_ids

    candidate_rows = 0
    candidate_source1_ids = set()

    empty_candidate_rows = 0
    total_candidate_ids = 0
    duplicate_candidate_rows = 0
    invalid_candidate_ids = set()

    print("Checking candidate_pairs.tsv...")

    with open(
        candidate_path,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f, delimiter="\t")

        columns = set(reader.fieldnames or [])

        missing_columns = REQUIRED_COLUMNS - columns

        if missing_columns:
            errors.append(
                "Missing required columns: "
                + ", ".join(sorted(missing_columns))
            )
        else:

            for line_number, row in enumerate(reader, start=2):

                candidate_rows += 1

                source1_id = (
                    row.get("source1_entity_id") or ""
                ).strip()

                candidate_string = (
                    row.get("candidate_entity_ids") or ""
                ).strip()

                # Check Source1 ID
                if not source1_id:
                    errors.append(
                        f"Line {line_number}: "
                        "missing source1_entity_id"
                    )
                    continue

                # Check duplicate Source1 rows
                if source1_id in candidate_source1_ids:
                    errors.append(
                        f"Line {line_number}: "
                        f"duplicate source1_entity_id: {source1_id}"
                    )

                candidate_source1_ids.add(source1_id)

                # Check Source1 exists
                if source1_id not in source1_ids:
                    errors.append(
                        f"Line {line_number}: "
                        f"unknown Source1 entity: {source1_id}"
                    )

                # Empty candidate list is allowed
                if not candidate_string:
                    empty_candidate_rows += 1
                    continue

                candidates = [
                    candidate.strip()
                    for candidate in candidate_string.split(",")
                    if candidate.strip()
                ]

                total_candidate_ids += len(candidates)

                # Duplicate candidate IDs inside one row
                if len(candidates) != len(set(candidates)):
                    duplicate_candidate_rows += 1

                    errors.append(
                        f"Line {line_number}: "
                        "duplicate candidate ID in row"
                    )

                # Check candidate IDs against Source2 + Source3
                for candidate_id in candidates:
                    if candidate_id not in valid_candidate_ids:
                        invalid_candidate_ids.add(candidate_id)

                        errors.append(
                            f"Line {line_number}: "
                            f"invalid candidate ID: {candidate_id}"
                        )

    # Check Source1 coverage
    missing_source1 = source1_ids - candidate_source1_ids
    extra_source1 = candidate_source1_ids - source1_ids

    if candidate_rows != len(source1_ids):
        errors.append(
            "Row count mismatch: "
            f"candidate_pairs={candidate_rows}, "
            f"test_source1={len(source1_ids)}"
        )

    if missing_source1:
        errors.append(
            f"Missing Source1 rows: {len(missing_source1)}"
        )

    if extra_source1:
        errors.append(
            f"Unexpected Source1 rows: {len(extra_source1)}"
        )

    # Summary
    print()
    print("=" * 60)
    print("QA VALIDATION SUMMARY")
    print("=" * 60)

    print(f"Test Source1 entities       : {len(source1_ids):,}")
    print(f"Candidate-pairs rows        : {candidate_rows:,}")
    print(
        f"Unique Source1 IDs          : "
        f"{len(candidate_source1_ids):,}"
    )
    print(
        f"Empty candidate rows        : "
        f"{empty_candidate_rows:,}"
    )
    print(
        f"Total candidate IDs         : "
        f"{total_candidate_ids:,}"
    )
    print(
        f"Duplicate candidate rows    : "
        f"{duplicate_candidate_rows:,}"
    )
    print(
        f"Invalid candidate IDs       : "
        f"{len(invalid_candidate_ids):,}"
    )
    print(
        f"Missing Source1 rows        : "
        f"{len(missing_source1):,}"
    )
    print(
        f"Unexpected Source1 rows     : "
        f"{len(extra_source1):,}"
    )

    print()
    print("=" * 60)

    if errors:
        print("FAIL")
        print("=" * 60)

        print()
        print(f"Problems found: {len(errors):,}")

        for error in errors[:20]:
            print(f"- {error}")

        if len(errors) > 20:
            print(
                f"... and {len(errors) - 20:,} more"
            )

        return False

    print("PASS")
    print("=" * 60)

    return True


def main():

    parser = argparse.ArgumentParser(
        description="Validate candidate_pairs.tsv"
    )

    parser.add_argument(
        "candidate_pairs",
        help="Path to candidate_pairs.tsv"
    )

    parser.add_argument(
        "--test-dir",
        default=(
            r"C:\Users\bhama\Downloads"
            r"\6ab10eb3b23ba_student_resource"
            r"\student_resource\dataset\test"
        ),
        help="Path to test dataset directory"
    )

    args = parser.parse_args()

    try:
        passed = validate(
            args.candidate_pairs,
            args.test_dir
        )

    except Exception as exc:
        print()
        print("=" * 60)
        print("FAIL")
        print("=" * 60)
        print(f"Error: {exc}")
        sys.exit(1)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()