import pandas as pd
from collections import defaultdict

from src.blocking.candidate_generator import create_blocking_keys


N = 10_000

print("Loading Source 1...")

s1 = pd.read_csv(
    "data/train/train_source1.tsv",
    sep="\t",
    dtype=str,
    nrows=N,
    keep_default_na=False
)

s1_ids = set(s1["entity_id"])

print("Loading ground truth...")

gt = pd.read_csv(
    "data/train/train_ground_truth.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

# Keep only ground-truth rows belonging to our 10,000 S1 records
gt = gt[gt["source1_entity_id"].isin(s1_ids)]

print("Ground-truth rows for sample:", len(gt))

print("Loading Source 2...")

s2 = pd.read_csv(
    "data/train/train_source2.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print("Loading Source 3...")

s3 = pd.read_csv(
    "data/train/train_source3.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

# ---------------------------------------------------------
# Build candidate blocking index
# ---------------------------------------------------------

print("Building blocking index...")

blocking_index = defaultdict(set)

for _, row in pd.concat([s2, s3], ignore_index=True).iterrows():

    for key in create_blocking_keys(row):
        blocking_index[key].add(row["entity_id"])

print("Unique blocking keys:", len(blocking_index))


# ---------------------------------------------------------
# Key frequency
# ---------------------------------------------------------

key_frequency = {
    key: len(ids)
    for key, ids in blocking_index.items()
}

MAX_FREQUENCY = 100

print("Frequency threshold:", MAX_FREQUENCY)


# ---------------------------------------------------------
# Ground truth dictionary
# ---------------------------------------------------------

ground_truth = {}

for _, row in gt.iterrows():

    value = row["matched_entity_ids"]

    matches = {
        x.strip()
        for x in value.split(",")
        if x.strip()
    }

    ground_truth[row["source1_entity_id"]] = matches


# ---------------------------------------------------------
# Evaluate blocking
# ---------------------------------------------------------

total_true_links = 0
captured_true_links = 0

matched_entities = 0
all_matches_captured = 0

candidate_counts = []


for _, row in s1.iterrows():

    s1_id = row["entity_id"]

    true_matches = ground_truth.get(s1_id, set())

    if true_matches:
        matched_entities += 1

    total_true_links += len(true_matches)

    candidate_ids = set()

    for key in create_blocking_keys(row):

        if key_frequency.get(key, 0) > MAX_FREQUENCY:
            continue

        candidate_ids.update(
            blocking_index.get(key, set())
        )

    candidate_counts.append(len(candidate_ids))

    captured = true_matches & candidate_ids

    captured_true_links += len(captured)

    if true_matches and captured == true_matches:
        all_matches_captured += 1


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print()
print("=" * 60)
print("CORRECTED BLOCKING RECALL RESULTS")
print("=" * 60)

print("Source 1 records tested:", N)

print("Matched Source 1 entities:", matched_entities)

print("Total true links:", total_true_links)

print("Captured true links:", captured_true_links)

if total_true_links:
    recall = captured_true_links / total_true_links

    print(
        "Blocking link recall:",
        round(recall * 100, 2),
        "%"
    )

if matched_entities:

    entity_recall = (
        all_matches_captured / matched_entities
    )

    print(
        "Entities with ALL matches captured:",
        all_matches_captured
    )

    print(
        "All-match entity recall:",
        round(entity_recall * 100, 2),
        "%"
    )

if candidate_counts:

    candidate_counts.sort()

    print()
    print("Candidate statistics:")

    print(
        "Average candidates:",
        round(
            sum(candidate_counts) / len(candidate_counts),
            2
        )
    )

    print(
        "Median candidates:",
        candidate_counts[len(candidate_counts) // 2]
    )

    print(
        "Maximum candidates:",
        candidate_counts[-1]
    )

print("=" * 60)