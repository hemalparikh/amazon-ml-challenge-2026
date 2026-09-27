import argparse
import csv
import re
from collections import defaultdict

from rapidfuzz import fuzz


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):
    if value is None:
        return ""

    value = str(value).lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def normalize_name(value):
    return normalize_text(value)


def normalize_address(value):
    return normalize_text(value)


def extract_address_number(value):
    """
    Extract the first numeric address component.

    Examples:
        '2621 Cotten Road' -> '2621'
        '45A MG Road'      -> '45a'
    """
    value = normalize_address(value)

    match = re.search(r"\b\d+[a-z]?\b", value)

    if match:
        return match.group(0)

    return ""


# ============================================================
# LOAD SOURCE FILE
# ============================================================

def load_source_file(path):
    """
    Load a source TSV into memory.

    Used for Source1, Source2 and Source3.
    """

    records = {}

    print(f"Loading: {path}")

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file,
            delimiter="\t",
        )

        required_columns = {
            "entity_id",
            "business_name",
            "business_address",
            "country",
        }

        actual_columns = set(reader.fieldnames or [])

        missing = required_columns - actual_columns

        if missing:
            raise ValueError(
                f"{path} is missing columns: {sorted(missing)}"
            )

        count = 0

        for row in reader:

            entity_id = (
                row.get("entity_id") or ""
            ).strip()

            if entity_id:
                records[entity_id] = row
                count += 1

    print(f"Loaded {count:,} records.")

    return records


# ============================================================
# CALCULATE SIMILARITIES
# ============================================================

def calculate_similarities(source1, candidate):

    name1 = normalize_name(
        source1.get("business_name")
    )

    name2 = normalize_name(
        candidate.get("business_name")
    )

    address1 = normalize_address(
        source1.get("business_address")
    )

    address2 = normalize_address(
        candidate.get("business_address")
    )

    country1 = normalize_text(
        source1.get("country")
    )

    country2 = normalize_text(
        candidate.get("country")
    )

    name_similarity = (
        fuzz.ratio(name1, name2) / 100.0
    )

    address_similarity = (
        fuzz.ratio(address1, address2) / 100.0
    )

    name_token_similarity = (
        fuzz.token_set_ratio(name1, name2) / 100.0
    )

    address_number1 = extract_address_number(
        address1
    )

    address_number2 = extract_address_number(
        address2
    )

    return {
        "name_similarity": name_similarity,
        "address_similarity": address_similarity,
        "name_token_similarity": name_token_similarity,

        "country_match": (
            bool(country1)
            and bool(country2)
            and country1 == country2
        ),

        "address_number_match": (
            bool(address_number1)
            and bool(address_number2)
            and address_number1 == address_number2
        ),

        "normalized_name1": name1,
        "normalized_name2": name2,
    }


# ============================================================
# RULES A-E
# ============================================================

def apply_rules(
    source1,
    candidate,
    name_threshold=0.90,
    address_threshold=0.80,
    strong_address_threshold=0.90,
    strong_name_threshold=0.95,
    combined_address_threshold=0.85,
):
    """
    Rule A:
        normalized_name == normalized_name

    Rule B:
        name similarity >= 0.90
        AND address similarity >= 0.80

    Rule C:
        address similarity >= 0.90
        AND address number matches

    Rule D:
        name token similarity >= 0.95
        AND country matches

    Rule E:
        name similarity >= 0.90
        AND address similarity >= 0.85
    """

    scores = calculate_similarities(
        source1,
        candidate,
    )

    rules = []

    name1 = scores["normalized_name1"]
    name2 = scores["normalized_name2"]

    name_similarity = scores[
        "name_similarity"
    ]

    address_similarity = scores[
        "address_similarity"
    ]

    name_token_similarity = scores[
        "name_token_similarity"
    ]

    # --------------------------------------------------------
    # Rule A — Exact normalized name
    # --------------------------------------------------------

    if (
        name1
        and name1 == name2
    ):
        rules.append("A")

    # --------------------------------------------------------
    # Rule B — Name + address
    # --------------------------------------------------------

    if (
        name_similarity >= name_threshold
        and address_similarity >= address_threshold
    ):
        rules.append("B")

    # --------------------------------------------------------
    # Rule C — Strong address
    # --------------------------------------------------------

    if (
        address_similarity >= strong_address_threshold
        and scores["address_number_match"]
    ):
        rules.append("C")

    # --------------------------------------------------------
    # Rule D — Strong name
    # --------------------------------------------------------

    if (
        name_token_similarity >= strong_name_threshold
        and scores["country_match"]
    ):
        rules.append("D")

    # --------------------------------------------------------
    # Rule E — Combined
    # --------------------------------------------------------

    if (
        name_similarity >= name_threshold
        and address_similarity >= combined_address_threshold
    ):
        rules.append("E")

    return rules, scores


# ============================================================
# STREAMING CANDIDATE PROCESSING
# ============================================================

def process_candidate_pairs(
    candidate_path,
    source1_records,
    source2_records,
    source3_records,
    name_threshold,
    address_threshold,
    strong_address_threshold,
    strong_name_threshold,
    combined_address_threshold,
):
    """
    Process candidate_pairs.tsv line-by-line.

    IMPORTANT:
    The 1.47 GB candidate file is NEVER loaded completely
    into memory.
    """

    # For each Source1 ID:
    # candidate_id -> evidence
    matches = defaultdict(dict)

    total_rows = 0
    valid_candidates = 0

    rule_counts = {
        "A": 0,
        "B": 0,
        "C": 0,
        "D": 0,
        "E": 0,
    }

    print()
    print("=" * 60)
    print("PROCESSING CANDIDATE PAIRS")
    print("=" * 60)

    with open(
        candidate_path,
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file,
            delimiter="\t",
        )

        if reader.fieldnames is None:
            raise ValueError(
                "Candidate file has no header."
            )

        columns = set(reader.fieldnames)

        required_columns = {
            "source1_entity_id",
            "candidate_entity_id",
        }

        missing = required_columns - columns

        if missing:
            raise ValueError(
                "Candidate file must contain: "
                "source1_entity_id and "
                f"candidate_entity_id. Missing: {sorted(missing)}"
            )

        for row in reader:

            total_rows += 1

            source1_id = (
                row.get("source1_entity_id") or ""
            ).strip()

            candidate_id = (
                row.get("candidate_entity_id") or ""
            ).strip()

            if (
                not source1_id
                or not candidate_id
            ):
                continue

            source1 = source1_records.get(
                source1_id
            )

            if source1 is None:
                continue

            # Search candidate in Source2 first
            candidate = source2_records.get(
                candidate_id
            )

            # Then Source3
            if candidate is None:
                candidate = source3_records.get(
                    candidate_id
                )

            if candidate is None:
                continue

            valid_candidates += 1

            rules, scores = apply_rules(
                source1,
                candidate,
                name_threshold=name_threshold,
                address_threshold=address_threshold,
                strong_address_threshold=strong_address_threshold,
                strong_name_threshold=strong_name_threshold,
                combined_address_threshold=combined_address_threshold,
            )

            # No rule passed
            if not rules:
                continue

            for rule in rules:
                rule_counts[rule] += 1

            # Evidence strength:
            # 1. Number of rules passed
            # 2. Name similarity
            # 3. Address similarity

            evidence = (
                len(rules),
                scores["name_similarity"],
                scores["address_similarity"],
                rules,
            )

            previous = matches[
                source1_id
            ].get(candidate_id)

            # Keep strongest evidence if duplicate
            # candidate pair occurs.
            if (
                previous is None
                or evidence[:3] > previous[:3]
            ):
                matches[
                    source1_id
                ][candidate_id] = evidence

            if total_rows % 5_000_000 == 0:

                print(
                    f"Candidate rows processed: "
                    f"{total_rows:,}"
                )

                print(
                    f"Valid candidate rows: "
                    f"{valid_candidates:,}"
                )

    # ========================================================
    # RANK FINAL MATCHES
    # ========================================================

    final_results = {}

    for source1_id in source1_records:

        candidate_matches = matches.get(
            source1_id,
            {}
        )

        ranked = sorted(
            candidate_matches.items(),
            key=lambda item: (
                -item[1][0],  # rule count
                -item[1][1],  # name similarity
                -item[1][2],  # address similarity
                item[0],
            ),
        )

        final_results[source1_id] = [
            candidate_id
            for candidate_id, evidence
            in ranked
        ]

    total_matches = sum(
        len(ids)
        for ids in final_results.values()
    )

    print()
    print("=" * 60)
    print("MATCHING SUMMARY")
    print("=" * 60)

    print(
        f"Candidate rows processed: "
        f"{total_rows:,}"
    )

    print(
        f"Valid candidate rows:     "
        f"{valid_candidates:,}"
    )

    print(
        f"Final matched links:       "
        f"{total_matches:,}"
    )

    print()
    print("Rule hits:")

    for rule, count in rule_counts.items():
        print(
            f"  Rule {rule}: {count:,}"
        )

    return final_results


# ============================================================
# WRITE FINAL OUTPUT
# ============================================================

def write_results(
    output_path,
    source1_records,
    final_results,
):
    """
    Write:

        source1_entity_id    matched_entity_ids

    One row for every Source1 ID.
    """

    print()
    print(
        f"Writing final output: {output_path}"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.writer(
            file,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writerow(
            [
                "source1_entity_id",
                "matched_entity_ids",
            ]
        )

        for source1_id in source1_records:

            matched_ids = final_results.get(
                source1_id,
                [],
            )

            writer.writerow(
                [
                    source1_id,
                    ",".join(matched_ids),
                ]
            )

    print("Output written successfully.")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Rule-based ensemble matcher "
            "for Amazon ML Challenge 2026"
        )
    )

    parser.add_argument(
        "--source1",
        required=True,
        help="Path to test_source1.tsv",
    )

    parser.add_argument(
        "--source2",
        required=True,
        help="Path to test_source2.tsv",
    )

    parser.add_argument(
        "--source3",
        required=True,
        help="Path to test_source3.tsv",
    )

    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidate_pairs.tsv",
    )

    parser.add_argument(
        "--output",
        default="matching_results_divya.tsv",
        help="Output TSV file",
    )

    # ========================================================
    # DEFAULT THRESHOLDS FROM TEAM LEAD
    # ========================================================

    parser.add_argument(
        "--name-threshold",
        type=float,
        default=0.90,
    )

    parser.add_argument(
        "--address-threshold",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--strong-address-threshold",
        type=float,
        default=0.90,
    )

    parser.add_argument(
        "--strong-name-threshold",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--combined-address-threshold",
        type=float,
        default=0.85,
    )

    args = parser.parse_args()

    print("=" * 60)
    print("AMAZON ML CHALLENGE")
    print("RULE-BASED ENSEMBLE MATCHER")
    print("=" * 60)

    print()
    print("Threshold configuration:")
    print(
        f"Name similarity:       "
        f"{args.name_threshold}"
    )
    print(
        f"Address similarity:    "
        f"{args.address_threshold}"
    )
    print(
        f"Strong address:        "
        f"{args.strong_address_threshold}"
    )
    print(
        f"Strong name:           "
        f"{args.strong_name_threshold}"
    )
    print(
        f"Combined address:      "
        f"{args.combined_address_threshold}"
    )

    # ========================================================
    # LOAD SOURCE FILES
    # ========================================================

    source1_records = load_source_file(
        args.source1
    )

    source2_records = load_source_file(
        args.source2
    )

    source3_records = load_source_file(
        args.source3
    )

    # ========================================================
    # PROCESS CANDIDATES
    # ========================================================

    final_results = process_candidate_pairs(
        candidate_path=args.candidates,
        source1_records=source1_records,
        source2_records=source2_records,
        source3_records=source3_records,
        name_threshold=args.name_threshold,
        address_threshold=args.address_threshold,
        strong_address_threshold=args.strong_address_threshold,
        strong_name_threshold=args.strong_name_threshold,
        combined_address_threshold=args.combined_address_threshold,
    )

    # ========================================================
    # WRITE RESULTS
    # ========================================================

    write_results(
        output_path=args.output,
        source1_records=source1_records,
        final_results=final_results,
    )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(
        f"Final file: {args.output}"
    )


if __name__ == "__main__":
    main()