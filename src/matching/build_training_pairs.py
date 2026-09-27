from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import os
import sys
import pandas as pd

from src.preprocessing.preprocess import preprocess_data
from src.blocking.candidate_generator import (
    create_blocking_keys,
    normalize_for_blocking,
)


TRAIN_DIR = "data/train"
OUTPUT_DIR = "data/matching"

S1_LIMIT = 10_000
MAX_KEY_FREQUENCY = 100
NEGATIVE_PER_S1 = 10


def load_ground_truth():
    path = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

    gt = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    truth = {}

    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        value = row["matched_entity_ids"]

        if not value:
            truth[s1_id] = set()
        else:
            truth[s1_id] = set(
                x.strip()
                for x in value.split(",")
                if x.strip()
            )

    return truth


def build_candidate_index(source2, source3):
    """
    Build a blocking index over Source2 + Source3.

    key -> set(candidate_entity_id)
    """

    index = {}

    combined = pd.concat(
        [
            source2,
            source3,
        ],
        ignore_index=True,
    )

    for _, row in combined.iterrows():
        keys = create_blocking_keys(row)

        for key in keys:
            bucket = index.setdefault(key, set())

            if len(bucket) < MAX_KEY_FREQUENCY:
                bucket.add(row["entity_id"])

    return index


def generate_candidates_for_row(row, index):
    """
    Generate candidate entity IDs for one Source1 row.
    """

    keys = create_blocking_keys(row)

    candidates = set()

    for key in keys:
        values = index.get(key)

        if values:
            candidates.update(values)

    return candidates


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("BUILDING REAL TRAINING PAIRS")
    print("=" * 60)

    print("\n[1/5] Loading Source1 subset...")

    s1 = pd.read_csv(
        os.path.join(TRAIN_DIR, "train_source1.tsv"),
        sep="\t",
        dtype=str,
        nrows=S1_LIMIT,
        keep_default_na=False,
    )

    print(f"Source1 rows: {len(s1):,}")

    print("\n[2/5] Loading Source2 and Source3...")

    s2 = pd.read_csv(
        os.path.join(TRAIN_DIR, "train_source2.tsv"),
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    s3 = pd.read_csv(
        os.path.join(TRAIN_DIR, "train_source3.tsv"),
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(f"Source2 rows: {len(s2):,}")
    print(f"Source3 rows: {len(s3):,}")

    print("\n[3/5] Preprocessing data...")

    s1 = preprocess_data(s1)
    s2 = preprocess_data(s2)
    s3 = preprocess_data(s3)

    print("Preprocessing complete.")

    print("\n[4/5] Building blocking index...")

    index = build_candidate_index(s2, s3)

    print(f"Blocking keys: {len(index):,}")

    print("\n[5/5] Generating labeled candidate pairs...")

    ground_truth = load_ground_truth()

    # Only keep ground truth for our S1 subset.
    subset_ids = set(s1["entity_id"])

    ground_truth = {
        k: v
        for k, v in ground_truth.items()
        if k in subset_ids
    }

    # Combined candidate lookup.
    candidates_by_id = {}

    for _, row in pd.concat([s2, s3], ignore_index=True).iterrows():
        candidates_by_id[row["entity_id"]] = row

    records = []

    total_candidates = 0
    positive_count = 0
    negative_count = 0
    missed_positive_count = 0

    for i, (_, row) in enumerate(s1.iterrows()):

        s1_id = row["entity_id"]

        true_matches = ground_truth.get(
            s1_id,
            set(),
        )

        candidates = generate_candidates_for_row(
            row,
            index,
        )

        # Always add all true matches.
        # This prevents blocking misses from
        # disappearing from the positive training data.
        candidates.update(true_matches)

        # Separate positives and negatives.
        positives = [
            cid
            for cid in candidates
            if cid in true_matches
        ]

        negatives = [
            cid
            for cid in candidates
            if cid not in true_matches
        ]

        # Keep the negative sample manageable.
        negatives = negatives[:NEGATIVE_PER_S1]

        for cid in positives:
            candidate_row = candidates_by_id.get(cid)

            if candidate_row is None:
                continue

            records.append(
                {
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": cid,
                    "source1_name": row["business_name"],
                    "candidate_name": candidate_row["business_name"],
                    "source1_address": row["business_address"],
                    "candidate_address": candidate_row["business_address"],
                    "source1_country": row["country"],
                    "candidate_country": candidate_row["country"],
                    "label": 1,
                }
            )

            positive_count += 1

        for cid in negatives:
            candidate_row = candidates_by_id.get(cid)

            if candidate_row is None:
                continue

            records.append(
                {
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": cid,
                    "source1_name": row["business_name"],
                    "candidate_name": candidate_row["business_name"],
                    "source1_address": row["business_address"],
                    "candidate_address": candidate_row["business_address"],
                    "source1_country": row["country"],
                    "candidate_country": candidate_row["country"],
                    "label": 0,
                }
            )

            negative_count += 1

        total_candidates += len(candidates)

        if i > 0 and i % 1000 == 0:
            print(
                f"Processed {i:,}/{len(s1):,} "
                f"S1 rows | "
                f"positives={positive_count:,} | "
                f"negatives={negative_count:,}"
            )

    output = pd.DataFrame(records)

    output_path = os.path.join(
        OUTPUT_DIR,
        "training_pairs.csv",
    )

    output.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 60)
    print("TRAINING PAIRS COMPLETE")
    print("=" * 60)

    print(f"Output: {output_path}")
    print(f"Rows: {len(output):,}")
    print(f"Positive pairs: {positive_count:,}")
    print(f"Negative pairs: {negative_count:,}")
    print(
        f"Average candidates/S1: "
        f"{total_candidates / max(len(s1), 1):.2f}"
    )


if __name__ == "__main__":
    main()