import pandas as pd
from collections import defaultdict

from src.blocking.candidate_generator import create_blocking_keys


N = 10_000
MAX_FREQUENCY = 100

print("Loading data...")

s1 = pd.read_csv(
    "data/train/train_source1.tsv",
    sep="\t",
    dtype=str,
    nrows=N,
    keep_default_na=False
)

gt = pd.read_csv(
    "data/train/train_ground_truth.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

s1_ids = set(s1["entity_id"])
gt = gt[gt["source1_entity_id"].isin(s1_ids)]

ground_truth = {}

for _, row in gt.iterrows():
    ground_truth[row["source1_entity_id"]] = {
        x.strip()
        for x in row["matched_entity_ids"].split(",")
        if x.strip()
    }

print("Loading Source 2 + Source 3...")

s2 = pd.read_csv(
    "data/train/train_source2.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

s3 = pd.read_csv(
    "data/train/train_source3.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

# ---------------------------------------------------------
# Build separate indexes for each key family
# ---------------------------------------------------------

indexes = {
    "NAME": defaultdict(set),
    "NAMELAST": defaultdict(set),
    "NUM": defaultdict(set),
    "ADDR": defaultdict(set),
}

print("Building indexes...")

for _, row in pd.concat([s2, s3], ignore_index=True).iterrows():

    for key in create_blocking_keys(row):

        parts = key.split("|")

        if len(parts) != 3:
            continue

        family = parts[1]

        if family in indexes:
            indexes[family][key].add(row["entity_id"])


# ---------------------------------------------------------
# Remove high-frequency keys
# ---------------------------------------------------------

for family in indexes:

    indexes[family] = {
        key: ids
        for key, ids in indexes[family].items()
        if len(ids) <= MAX_FREQUENCY
    }


# ---------------------------------------------------------
# Evaluate each family
# ---------------------------------------------------------

def evaluate(families):

    captured = 0
    total = 0

    candidate_counts = []

    for _, row in s1.iterrows():

        s1_id = row["entity_id"]

        true_matches = ground_truth.get(s1_id, set())

        total += len(true_matches)

        candidates = set()

        for family in families:

            for key in create_blocking_keys(row):

                if key.startswith(
                    tuple(f"{x}|" for x in [])
                ):
                    pass

                parts = key.split("|")

                if len(parts) != 3:
                    continue

                if parts[1] == family:
                    candidates.update(
                        indexes[family].get(key, set())
                    )

        candidate_counts.append(len(candidates))

        captured += len(true_matches & candidates)

    recall = captured / total if total else 0

    return (
        recall * 100,
        sum(candidate_counts) / len(candidate_counts),
        sorted(candidate_counts)[len(candidate_counts) // 2],
        max(candidate_counts),
    )


print()
print("=" * 70)
print("BLOCKING FAMILY ANALYSIS")
print("=" * 70)

for families in [
    ["NAME"],
    ["NAMELAST"],
    ["NUM"],
    ["ADDR"],
    ["NAME", "NAMELAST"],
    ["NAME", "NUM"],
    ["NAME", "ADDR"],
    ["NAME", "NAMELAST", "NUM", "ADDR"],
]:

    recall, avg, median, maximum = evaluate(families)

    print(
        f"{'+'.join(families):25s}"
        f" Recall={recall:6.2f}%"
        f" AvgCandidates={avg:7.2f}"
        f" Median={median:4d}"
        f" Max={maximum:4d}"
    )

print("=" * 70)