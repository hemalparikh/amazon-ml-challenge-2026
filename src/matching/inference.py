import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.matching.features import create_features


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_SOURCE1 = "data/test/test_source1.tsv"
DEFAULT_SOURCE2 = "data/test/test_source2.tsv"
DEFAULT_SOURCE3 = "data/test/test_source3.tsv"

DEFAULT_CANDIDATES = "data/matching/candidate_pairs.tsv"
DEFAULT_MODEL = "models/matching_model_real.pkl"

DEFAULT_RESULTS = "data/matching/matching_results.tsv"
DEFAULT_CANDIDATE_OUTPUT = "data/matching/candidate_pairs.tsv"


EXPECTED_FEATURE_COUNT = 25
DEFAULT_THRESHOLD = 0.80


# ============================================================
# FILE LOADING
# ============================================================

def load_tsv(path):
    """
    Load a TSV file as strings.

    Raises:
        FileNotFoundError: if the file does not exist.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Input path is not a file: {path}"
        )

    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
    )


# ============================================================
# COLUMN HELPERS
# ============================================================

def find_column(df, candidates, description):
    """
    Find a column using case-insensitive matching.
    """

    lower_map = {
        str(column).lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    raise ValueError(
        f"Could not find {description} column. "
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# CANDIDATE PAIR STANDARDIZATION
# ============================================================

def standardize_candidate_pairs(df):
    """
    Convert candidate-pair columns into the names expected
    by create_features().

    This function supports candidate files that already contain
    the full entity information.
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

    result = result.rename(
        columns=rename_map
    )

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


# ============================================================
# CANDIDATE DETAIL ENRICHMENT
# ============================================================

def attach_candidate_details(
    candidates,
    source1,
    source2,
    source3,
):
    """
    If candidate_pairs.tsv contains only IDs, enrich it using
    Source1/Source2/Source3 test files.

    This keeps the inference pipeline compatible with
    Divya's blocking output.
    """

    candidates = candidates.copy()

    # --------------------------------------------------------
    # SOURCE 1
    # --------------------------------------------------------

    source1_id_col = find_column(
        source1,
        [
            "source1_entity_id",
            "entity_id",
            "id",
        ],
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
            column = find_column(
                source1,
                possible,
                f"Source1 {field}",
            )

            source1_columns[column] = (
                f"source1_{field}"
            )

        except ValueError:
            pass

    s1 = source1.rename(
        columns=source1_columns
    ).copy()

    s1_keep = [
        column
        for column in [
            "source1_entity_id",
            "source1_name",
            "source1_address",
            "source1_country",
        ]
        if column in s1.columns
    ]

    s1 = (
        s1[s1_keep]
        .drop_duplicates(
            "source1_entity_id"
        )
    )

    # --------------------------------------------------------
    # SOURCE 2 / SOURCE 3
    # --------------------------------------------------------

    def prepare_candidate_source(
        df,
        prefix,
    ):
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

        for field in [
            "name",
            "address",
            "country",
        ]:

            possible = [
                f"{prefix}_{field}",
                field,
                f"entity_{field}",
            ]

            try:
                column = find_column(
                    df,
                    possible,
                    f"{prefix} {field}",
                )

                rename[column] = (
                    f"candidate_{field}"
                )

            except ValueError:
                pass

        out = df.rename(
            columns=rename
        ).copy()

        for field in [
            "name",
            "address",
            "country",
        ]:

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

    s2 = prepare_candidate_source(
        source2,
        "source2",
    )

    s3 = prepare_candidate_source(
        source3,
        "source3",
    )

    candidate_lookup = pd.concat(
        [
            s2,
            s3,
        ],
        ignore_index=True,
    )

    candidate_lookup = (
        candidate_lookup
        .drop_duplicates(
            "candidate_entity_id"
        )
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


# ============================================================
# RESULT GENERATION
# ============================================================

def generate_results(
    candidates,
    probabilities,
    source1,
    threshold,
):
    """
    Generate exactly one output row per Source1 entity.

    Only candidate pairs can become final matches.
    """

    candidates = candidates.copy()

    candidates["match_probability"] = (
        probabilities
    )

    candidates["predicted_match"] = (
        candidates["match_probability"]
        >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # ONLY CANDIDATE PAIRS CAN BECOME MATCHES
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
        .groupby(
            "source1_entity_id"
        )["candidate_entity_id"]
        .apply(
            lambda values:
            ",".join(
                dict.fromkeys(
                    values.astype(str)
                )
            )
        )
    )

    source1_id_col = find_column(
        source1,
        [
            "source1_entity_id",
            "entity_id",
            "id",
        ],
        "Source1 entity ID",
    )

    all_source1 = (
        source1[
            [source1_id_col]
        ]
        .rename(
            columns={
                source1_id_col:
                "source1_entity_id"
            }
        )
        .drop_duplicates()
    )

    all_source1[
        "matched_entity_ids"
    ] = (
        all_source1[
            "source1_entity_id"
        ]
        .map(grouped)
        .fillna("")
    )

    return all_source1[
        [
            "source1_entity_id",
            "matched_entity_ids",
        ]
    ]


# ============================================================
# OUTPUT VALIDATION
# ============================================================

def validate_results(
    results,
    candidates,
    source1,
):
    """
    Validate production output.

    Checks:
    - Every Source1 entity appears exactly once.
    - No duplicate matched IDs.
    - Every final matched pair exists in candidate pairs.
    - Empty candidate lists remain empty.
    """

    # --------------------------------------------------------
    # SOURCE1 ID VALIDATION
    # --------------------------------------------------------

    source1_id_col = find_column(
        source1,
        [
            "source1_entity_id",
            "entity_id",
            "id",
        ],
        "Source1 entity ID",
    )

    source1_ids = (
        source1[source1_id_col]
        .astype(str)
        .drop_duplicates()
    )

    source1_id_set = set(
        source1_ids
    )

    result_ids = (
        results["source1_entity_id"]
        .astype(str)
    )

    result_id_set = set(
        result_ids
    )

    if source1_id_set != result_id_set:
        missing = (
            source1_id_set
            - result_id_set
        )

        extra = (
            result_id_set
            - source1_id_set
        )

        raise ValueError(
            "Output Source1 IDs do not exactly "
            "match test Source1 IDs. "
            f"Missing={len(missing)}, "
            f"Extra={len(extra)}"
        )

    # --------------------------------------------------------
    # EXACTLY ONE OUTPUT ROW PER SOURCE1
    # --------------------------------------------------------

    if len(results) != len(source1_id_set):
        raise ValueError(
            "Every Source1 entity must appear "
            "exactly once."
        )

    if results[
        "source1_entity_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate Source1 IDs found "
            "in final output."
        )

    # --------------------------------------------------------
    # CANDIDATE PAIR SET
    # --------------------------------------------------------

    candidate_set = set(
        zip(
            candidates[
                "source1_entity_id"
            ].astype(str),

            candidates[
                "candidate_entity_id"
            ].astype(str),
        )
    )

    # --------------------------------------------------------
    # VALIDATE FINAL MATCH IDS
    # --------------------------------------------------------

    for _, row in results.iterrows():

        source1_id = str(
            row["source1_entity_id"]
        )

        matched_ids = str(
            row["matched_entity_ids"]
        ).strip()

        # Empty match list is valid.
        if not matched_ids:
            continue

        ids = [
            item.strip()
            for item in matched_ids.split(",")
            if item.strip()
        ]

        # ----------------------------------------------------
        # NO DUPLICATED OUTPUT IDs
        # ----------------------------------------------------

        if len(ids) != len(set(ids)):
            raise ValueError(
                f"Duplicate matched candidate IDs "
                f"for Source1 {source1_id}"
            )

        # ----------------------------------------------------
        # EVERY FINAL MATCH MUST BE A CANDIDATE
        # ----------------------------------------------------

        for candidate_id in ids:

            pair = (
                source1_id,
                candidate_id,
            )

            if pair not in candidate_set:
                raise ValueError(
                    "Invalid final match outside "
                    f"candidate pairs: {pair}"
                )

    print(
        "Output validation passed."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run entity matching inference pipeline."
        )
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
        default=DEFAULT_THRESHOLD,
        help=(
            "Matching probability threshold "
            "(default: 0.80)"
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # THRESHOLD VALIDATION
    # ========================================================

    if not 0.0 <= args.threshold <= 1.0:
        parser.error(
            "--threshold must be between 0.0 and 1.0"
        )

    print("=" * 70)
    print("ENTITY MATCHING INFERENCE")
    print("=" * 70)

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("\nLoading Source1...")
    source1 = load_tsv(
        args.source1
    )
    print(
        f"Source1 rows: {len(source1):,}"
    )

    print("\nLoading Source2...")
    source2 = load_tsv(
        args.source2
    )
    print(
        f"Source2 rows: {len(source2):,}"
    )

    print("\nLoading Source3...")
    source3 = load_tsv(
        args.source3
    )
    print(
        f"Source3 rows: {len(source3):,}"
    )

    print("\nLoading candidate pairs...")
    candidates = load_tsv(
        args.candidates
    )

    print(
        f"Candidate pairs: {len(candidates):,}"
    )

    # ========================================================
    # REQUIRED CANDIDATE COLUMNS
    # ========================================================

    required_candidate_columns = {
        "source1_entity_id",
        "candidate_entity_id",
    }

    missing_candidate_columns = (
        required_candidate_columns
        - set(candidates.columns)
    )

    if missing_candidate_columns:

        raise ValueError(
            "Candidate file is missing required "
            "columns: "
            + ", ".join(
                sorted(
                    missing_candidate_columns
                )
            )
        )

    # ========================================================
    # REMOVE DUPLICATE CANDIDATE PAIRS
    # ========================================================

    candidate_key = [
        "source1_entity_id",
        "candidate_entity_id",
    ]

    duplicate_pairs = (
        candidates
        .duplicated(
            subset=candidate_key
        )
        .sum()
    )

    if duplicate_pairs:

        print(
            "Warning: removing "
            f"{duplicate_pairs:,} "
            "duplicate candidate pairs."
        )

        candidates = (
            candidates
            .drop_duplicates(
                subset=candidate_key
            )
            .reset_index(
                drop=True
            )
        )

    print(
        "Unique candidate pairs: "
        f"{len(candidates):,}"
    )

    # ========================================================
    # SAVE / PRESERVE CANDIDATE PAIRS
    # ========================================================

    candidate_output = Path(
        args.candidate_output
    )

    candidate_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates.to_csv(
        candidate_output,
        sep="\t",
        index=False,
    )

    print(
        "Candidate pairs saved to: "
        f"{candidate_output}"
    )

    # ========================================================
    # ENRICH CANDIDATES
    # ========================================================

    print(
        "\nPreparing candidate data..."
    )

    candidates = attach_candidate_details(
        candidates,
        source1,
        source2,
        source3,
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print("\nLoading model...")

    bundle = joblib.load(
        args.model
    )

    if not isinstance(bundle, dict):
        raise TypeError(
            "Expected matching_model_real.pkl "
            "to contain a model dictionary."
        )

    required_model_keys = {
        "model",
        "feature_columns",
    }

    missing_model_keys = (
        required_model_keys
        - set(bundle.keys())
    )

    if missing_model_keys:
        raise ValueError(
            "Model bundle is missing required keys: "
            + ", ".join(
                sorted(
                    missing_model_keys
                )
            )
        )

    model = bundle["model"]

    feature_columns = bundle[
        "feature_columns"
    ]

    model_threshold = bundle.get(
        "threshold",
        DEFAULT_THRESHOLD,
    )

    print(
        f"Model: {type(model).__name__}"
    )

    print(
        f"Model threshold: "
        f"{model_threshold}"
    )

    print(
        f"Runtime threshold: "
        f"{args.threshold}"
    )

    print(
        f"Expected features: "
        f"{len(feature_columns)}"
    )

    # ========================================================
    # EXACT 25 FEATURE VALIDATION
    # ========================================================

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:

        raise ValueError(
            "Expected exactly "
            f"{EXPECTED_FEATURE_COUNT} "
            "model features, but model contains "
            f"{len(feature_columns)}"
        )

    model_feature_count = getattr(
        model,
        "n_features_in_",
        EXPECTED_FEATURE_COUNT,
    )

    if model_feature_count != EXPECTED_FEATURE_COUNT:

        raise ValueError(
            f"Model expects "
            f"{model_feature_count} features, "
            f"expected {EXPECTED_FEATURE_COUNT}"
        )

    # ========================================================
    # CREATE FEATURES
    # ========================================================

    print("\nCreating features...")

    features = create_features(
        candidates
    )

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in features.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing model features:\n"
            + "\n".join(
                missing_features
            )
        )

    X = features[
        feature_columns
    ].copy()

    # Convert everything to numeric.
    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0)

    print(
        f"Feature matrix shape: "
        f"{X.shape}"
    )

    if X.shape[1] != EXPECTED_FEATURE_COUNT:

        raise ValueError(
            "Feature count mismatch: "
            f"got {X.shape[1]}, "
            f"expected {EXPECTED_FEATURE_COUNT}"
        )

    # ========================================================
    # PREDICTION
    # ========================================================

    print(
        "\nRunning model prediction..."
    )

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    candidates[
        "match_probability"
    ] = probabilities

    predicted_matches = (
        probabilities
        >= args.threshold
    )

    print(
        "Predicted matches: "
        f"{predicted_matches.sum():,}"
    )

    if len(predicted_matches) > 0:

        print(
            "Match rate among candidate "
            "pairs: "
            f"{predicted_matches.mean():.4%}"
        )

    else:

        print(
            "Match rate among candidate "
            "pairs: 0.0000%"
        )

    # ========================================================
    # GENERATE FINAL RESULTS
    # ========================================================

    print(
        "\nGenerating matching_results.tsv..."
    )

    results = generate_results(
        candidates,
        probabilities,
        source1,
        args.threshold,
    )

    result_path = Path(
        args.results
    )

    result_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        result_path,
        sep="\t",
        index=False,
    )

    # ========================================================
    # VALIDATE OUTPUT
    # ========================================================

    print(
        "\nValidating output..."
    )

    validate_results(
        results,
        candidates,
        source1,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "INFERENCE COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Source1 entities:       "
        f"{len(source1):,}"
    )

    print(
        f"Candidate pairs:        "
        f"{len(candidates):,}"
    )

    print(
        f"Predicted match pairs:  "
        f"{predicted_matches.sum():,}"
    )

    print(
        f"Source1 output rows:     "
        f"{len(results):,}"
    )

    print(
        f"Threshold:              "
        f"{args.threshold}"
    )

    print(
        f"Results:                "
        f"{result_path}"
    )

    print(
        f"Candidates:             "
        f"{candidate_output}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()