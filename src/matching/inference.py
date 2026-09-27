import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.matching.features import create_features


DEFAULT_SOURCE1 = "data/test/test_source1.tsv"
DEFAULT_SOURCE2 = "data/test/test_source2.tsv"
DEFAULT_SOURCE3 = "data/test/test_source3.tsv"

DEFAULT_CANDIDATES = "data/matching/candidate_pairs.tsv"
DEFAULT_MODEL = "models/matching_model_real.pkl"

DEFAULT_RESULTS = "data/matching/matching_results.tsv"
DEFAULT_CANDIDATE_OUTPUT = "data/matching/candidate_pairs.tsv"


def load_tsv(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    return pd.read_csv(path, sep="\t", dtype=str)


def find_column(df, candidates, description):
    """
    Find a column using case-insensitive matching.
    """
    lower_map = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    raise ValueError(
        f"Could not find {description} column. "
        f"Available columns: {list(df.columns)}"
    )


def standardize_candidate_pairs(df):
    """
    Convert candidate-pair columns into the names expected
    by create_features().
    """

    source1_id = find_column(
        df,
        [
            "source1_entity_id",
            "source1_id",
            "entity_id",
            "source1",
        ],
        "Source1 entity ID",
    )

    candidate_id = find_column(
        df,
        [
            "candidate_entity_id",
            "candidate_id",
            "source2_entity_id",
            "source3_entity_id",
            "candidate",
        ],
        "candidate entity ID",
    )

    source1_name = find_column(
        df,
        ["source1_name"],
        "Source1 name",
    )

    source1_address = find_column(
        df,
        ["source1_address"],
        "Source1 address",
    )

    source1_country = find_column(
        df,
        ["source1_country"],
        "Source1 country",
    )

    candidate_name = find_column(
        df,
        ["candidate_name"],
        "candidate name",
    )

    candidate_address = find_column(
        df,
        ["candidate_address"],
        "candidate address",
    )

    candidate_country = find_column(
        df,
        ["candidate_country"],
        "candidate country",
    )

    result = df.copy()

    rename_map = {
        source1_id: "source1_entity_id",
        candidate_id: "candidate_entity_id",
        source1_name: "source1_name",
        source1_address: "source1_address",
        source1_country: "source1_country",
        candidate_name: "candidate_name",
        candidate_address: "candidate_address",
        candidate_country: "candidate_country",
    }

    result = result.rename(columns=rename_map)

    required = [
        "source1_entity_id",
        "candidate_entity_id",
        "source1_name",
        "source1_address",
        "source1_country",
        "candidate_name",
        "candidate_address",
        "candidate_country",
    ]

    for column in required:
        if column not in result.columns:
            result[column] = ""

    return result


def attach_candidate_details(
    candidates,
    source1,
    source2,
    source3,
):
    """
    If candidate_pairs.tsv contains only IDs, enrich it using
    Source1/Source2/Source3 test files.

    This keeps the inference pipeline flexible for Divya's
    blocking output.
    """

    candidates = candidates.copy()

    # --------------------------------------------------------
    # SOURCE 1
    # --------------------------------------------------------

    source1_id_col = find_column(
        source1,
        ["source1_entity_id", "entity_id", "id"],
        "Source1 entity ID",
    )

    source1_columns = {
        source1_id_col: "source1_entity_id"
    }

    for field in ["name", "address", "country"]:
        possible = [
            f"source1_{field}",
            field,
            f"entity_{field}",
        ]

        try:
            col = find_column(
                source1,
                possible,
                f"Source1 {field}",
            )
            source1_columns[col] = f"source1_{field}"
        except ValueError:
            source1_columns[col if False else source1_id_col] = source1_id_col

    s1 = source1.rename(columns=source1_columns).copy()

    s1_keep = [
        c for c in [
            "source1_entity_id",
            "source1_name",
            "source1_address",
            "source1_country",
        ]
        if c in s1.columns
    ]

    s1 = s1[s1_keep].drop_duplicates("source1_entity_id")

    # --------------------------------------------------------
    # SOURCE 2 / SOURCE 3
    # --------------------------------------------------------

    def prepare_candidate_source(df, prefix):
        id_col = find_column(
            df,
            [
                f"{prefix}_entity_id",
                "entity_id",
                "id",
            ],
            f"{prefix} entity ID",
        )

        rename = {
            id_col: "candidate_entity_id"
        }

        for field in ["name", "address", "country"]:
            possible = [
                f"{prefix}_{field}",
                field,
                f"entity_{field}",
            ]

            try:
                col = find_column(
                    df,
                    possible,
                    f"{prefix} {field}",
                )
                rename[col] = f"candidate_{field}"
            except ValueError:
                pass

        out = df.rename(columns=rename).copy()

        for field in ["name", "address", "country"]:
            column = f"candidate_{field}"

            if column not in out.columns:
                out[column] = ""

        return out[
            [
                "candidate_entity_id",
                "candidate_name",
                "candidate_address",
                "candidate_country",
            ]
        ]

    s2 = prepare_candidate_source(source2, "source2")
    s3 = prepare_candidate_source(source3, "source3")

    candidate_lookup = pd.concat(
        [s2, s3],
        ignore_index=True,
    )

    candidate_lookup = candidate_lookup.drop_duplicates(
        "candidate_entity_id"
    )

    # --------------------------------------------------------
    # ENRICH CANDIDATES
    # --------------------------------------------------------

    candidates = candidates.drop(
        columns=[
            "source1_name",
            "source1_address",
            "source1_country",
            "candidate_name",
            "candidate_address",
            "candidate_country",
        ],
        errors="ignore",
    )

    candidates = candidates.merge(
        s1,
        on="source1_entity_id",
        how="left",
    )

    candidates = candidates.merge(
        candidate_lookup,
        on="candidate_entity_id",
        how="left",
    )

    return candidates


def generate_results(
    candidates,
    probabilities,
    source1,
    threshold,
):
    """
    Generate exactly one output row per Source1 entity.
    """

    candidates = candidates.copy()

    candidates["match_probability"] = probabilities

    candidates["predicted_match"] = (
        candidates["match_probability"] >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # IMPORTANT:
    # Only candidate pairs can become final matches.
    # --------------------------------------------------------

    matches = candidates[
        candidates["predicted_match"] == 1
    ].copy()

    matches = matches[
        [
            "source1_entity_id",
            "candidate_entity_id",
        ]
    ]

    grouped = (
        matches
        .groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda x: ",".join(dict.fromkeys(x.astype(str))))
    )

    source1_ids = find_column(
        source1,
        [
            "source1_entity_id",
            "entity_id",
            "id",
        ],
        "Source1 entity ID",
    )

    all_source1 = (
        source1[[source1_ids]]
        .rename(columns={source1_ids: "source1_entity_id"})
        .drop_duplicates()
    )

    all_source1["matched_entity_ids"] = (
        all_source1["source1_entity_id"]
        .map(grouped)
        .fillna("")
    )

    return all_source1[
        [
            "source1_entity_id",
            "matched_entity_ids",
        ]
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Run entity matching inference pipeline."
    )

    parser.add_argument(
        "--source1",
        default=DEFAULT_SOURCE1,
    )

    parser.add_argument(
        "--source2",
        default=DEFAULT_SOURCE2,
    )

    parser.add_argument(
        "--source3",
        default=DEFAULT_SOURCE3,
    )

    parser.add_argument(
        "--candidates",
        default=DEFAULT_CANDIDATES,
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--results",
        default=DEFAULT_RESULTS,
    )

    parser.add_argument(
        "--candidate-output",
        default=DEFAULT_CANDIDATE_OUTPUT,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ENTITY MATCHING INFERENCE")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading Source1...")
    source1 = load_tsv(args.source1)
    print(f"Source1 rows: {len(source1):,}")

    print("\nLoading Source2...")
    source2 = load_tsv(args.source2)
    print(f"Source2 rows: {len(source2):,}")

    print("\nLoading Source3...")
    source3 = load_tsv(args.source3)
    print(f"Source3 rows: {len(source3):,}")

    print("\nLoading candidate pairs...")
    candidates = load_tsv(args.candidates)
    print(f"Candidate pairs: {len(candidates):,}")

    # --------------------------------------------------------
    # SAVE / PRESERVE CANDIDATE PAIRS
    # --------------------------------------------------------

    candidate_output = Path(args.candidate_output)
    candidate_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates.to_csv(
        candidate_output,
        sep="\t",
        index=False,
    )

    print(f"Candidate pairs saved to: {candidate_output}")

    # --------------------------------------------------------
    # ENRICH CANDIDATES
    # --------------------------------------------------------

    print("\nPreparing candidate data...")

    candidates = attach_candidate_details(
        candidates,
        source1,
        source2,
        source3,
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\nLoading model...")

    bundle = joblib.load(args.model)

    if not isinstance(bundle, dict):
        raise TypeError(
            "Expected matching_model_real.pkl to contain a model dictionary."
        )

    model = bundle["model"]

    feature_columns = bundle["feature_columns"]

    model_threshold = bundle.get(
        "threshold",
        0.80,
    )

    print(f"Model: {type(model).__name__}")
    print(f"Model threshold: {model_threshold}")
    print(f"Runtime threshold: {args.threshold}")
    print(f"Expected features: {len(feature_columns)}")

    # --------------------------------------------------------
    # CREATE FEATURES
    # --------------------------------------------------------

    print("\nCreating features...")

    features = create_features(candidates)

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in features.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing model features:\n"
            + "\n".join(missing_features)
        )

    X = features[feature_columns].copy()

    # Convert everything to numeric.
    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0)

    print(f"Feature matrix shape: {X.shape}")

    if X.shape[1] != len(feature_columns):
        raise ValueError(
            f"Feature count mismatch: "
            f"got {X.shape[1]}, "
            f"expected {len(feature_columns)}"
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    print("\nRunning model prediction...")

    probabilities = model.predict_proba(X)[:, 1]

    candidates["match_probability"] = probabilities

    predicted_matches = (
        probabilities >= args.threshold
    )

    print(
        f"Predicted matches: "
        f"{predicted_matches.sum():,}"
    )

    print(
        f"Match rate among candidate pairs: "
        f"{predicted_matches.mean():.4%}"
    )

    # --------------------------------------------------------
    # GENERATE FINAL RESULTS
    # --------------------------------------------------------

    print("\nGenerating matching_results.tsv...")

    results = generate_results(
        candidates,
        probabilities,
        source1,
        args.threshold,
    )

    result_path = Path(args.results)

    result_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        result_path,
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print("\nValidating output...")

    source1_ids = set(
        source1[
            find_column(
                source1,
                [
                    "source1_entity_id",
                    "entity_id",
                    "id",
                ],
                "Source1 entity ID",
            )
        ].astype(str)
    )

    result_ids = set(
        results["source1_entity_id"].astype(str)
    )

    if source1_ids != result_ids:
        raise ValueError(
            "Output Source1 IDs do not exactly match test Source1 IDs."
        )

    if len(results) != len(source1_ids):
        raise ValueError(
            "Every Source1 entity must appear exactly once."
        )

    # Verify every predicted pair belongs to candidates.
    candidate_set = set(
        zip(
            candidates["source1_entity_id"].astype(str),
            candidates["candidate_entity_id"].astype(str),
        )
    )

    for _, row in candidates[
        candidates["match_probability"] >= args.threshold
    ].iterrows():

        pair = (
            str(row["source1_entity_id"]),
            str(row["candidate_entity_id"]),
        )

        if pair not in candidate_set:
            raise ValueError(
                f"Invalid match outside candidate pairs: {pair}"
            )

    print("Output validation passed.")

    print("\n" + "=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)

    print(f"Source1 entities:       {len(source1):,}")
    print(f"Candidate pairs:       {len(candidates):,}")
    print(
        f"Predicted match pairs: "
        f"{predicted_matches.sum():,}"
    )
    print(
        f"Source1 output rows:   "
        f"{len(results):,}"
    )
    print(f"Threshold:             {args.threshold}")
    print(f"Results:               {result_path}")
    print(f"Candidates:            {candidate_output}")


if __name__ == "__main__":
    main()