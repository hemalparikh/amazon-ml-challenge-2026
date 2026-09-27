import argparse
from pathlib import Path

import pandas as pd

from src.matching.features import create_features


DEFAULT_CANDIDATES = "data/matching/candidate_pairs.tsv"
DEFAULT_OUTPUT = "data/matching/matching_results_abhishek.tsv"

WEIGHTS = {
    "name_similarity": 0.35,
    "name_token_set_similarity": 0.20,
    "address_similarity": 0.25,
    "address_token_set_similarity": 0.10,
    "country_exact_match": 0.10,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precision-focused fuzzy matcher over existing candidate pairs."
    )

    parser.add_argument(
        "--candidates",
        default=DEFAULT_CANDIDATES,
        help=f"Candidate pairs TSV (default: {DEFAULT_CANDIDATES})",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output TSV (default: {DEFAULT_OUTPUT})",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Matching score threshold between 0 and 1 (default: 0.90)",
    )

    return parser.parse_args()


def validate_threshold(threshold):
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"Threshold must be between 0.0 and 1.0, got {threshold}"
        )


def load_candidates(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Candidate file not found: {path}"
        )

    return pd.read_csv(path, sep="\t")


def validate_columns(df):
    required = {
        "source1_entity_id",
        "candidate_entity_id",
        "source1_name",
        "candidate_name",
        "source1_address",
        "candidate_address",
        "source1_country",
        "candidate_country",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Candidate file is missing required columns: "
            + ", ".join(sorted(missing))
        )


def calculate_weighted_score(features):
    return (
        features["name_similarity"] * WEIGHTS["name_similarity"]
        + features["name_token_set_similarity"]
        * WEIGHTS["name_token_set_similarity"]
        + features["address_similarity"]
        * WEIGHTS["address_similarity"]
        + features["address_token_set_similarity"]
        * WEIGHTS["address_token_set_similarity"]
        + features["country_exact_match"]
        * WEIGHTS["country_exact_match"]
    )


def build_results(features, threshold):
    features = features.copy()

    features["weighted_score"] = calculate_weighted_score(
        features
    )

    matches = features[
        features["weighted_score"] >= threshold
    ].copy()

    # Prevent duplicate candidate pairs from creating
    # duplicate IDs in the final output.
    matches = matches.drop_duplicates(
        subset=[
            "source1_entity_id",
            "candidate_entity_id",
        ]
    )

    results = (
        matches.groupby("source1_entity_id")[
            "candidate_entity_id"
        ]
        .apply(
            lambda values: ",".join(
                dict.fromkeys(
                    str(value)
                    for value in values
                    if pd.notna(value)
                )
            )
        )
        .reset_index()
    )

    results.columns = [
        "source1_entity_id",
        "matched_entity_ids",
    ]

    return results


def main():
    args = parse_args()

    validate_threshold(args.threshold)

    print("=" * 70)
    print("PRECISION-FOCUSED FUZZY MATCHER")
    print("=" * 70)

    print(f"\nCandidate file: {args.candidates}")
    print(f"Output file:    {args.output}")
    print(f"Threshold:      {args.threshold}")

    print("\nWeights:")
    for feature, weight in WEIGHTS.items():
        print(f"  {feature}: {weight}")

    print("\nLoading candidate pairs...")

    candidates = load_candidates(args.candidates)

    print(
        f"Candidate pairs loaded: {len(candidates):,}"
    )

    validate_columns(candidates)

    print("\nCreating normalized/fuzzy features...")

    features = create_features(candidates)

    required_features = [
        "name_similarity",
        "name_token_set_similarity",
        "address_similarity",
        "address_token_set_similarity",
        "country_exact_match",
    ]

    missing_features = [
        column
        for column in required_features
        if column not in features.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required fuzzy features: "
            + ", ".join(missing_features)
        )

    print("Calculating weighted scores...")

    results = build_results(
        features,
        args.threshold,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print("\n" + "=" * 70)
    print("MATCHING COMPLETE")
    print("=" * 70)

    print(
        f"Matched Source1 rows: {len(results):,}"
    )

    print(
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()