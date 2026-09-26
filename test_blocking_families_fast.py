import pandas as pd
from collections import defaultdict

from src.blocking.candidate_generator import create_blocking_keys


N = 10_000
MAX_FREQUENCY = 100

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

gt = gt[gt["source1_entity_id"].isin(s1_ids)]

ground_truth = dict(
    zip(
        gt["source1_entity_id"],
        gt["matched_entity_ids"].map(
            lambda x: {
                v.strip()
                for v in x.split(",")
                if v.strip()
            }
        )
    )
)

print("Ground-truth rows:", len(gt))

# ---------------------------------------------------------
# Load S2 + S3
# ---------------------------------------------------------

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

sources = pd.concat(
    [s2, s3],
    ignore_index=True
)

print("Total candidate records:", len(sources))

# ---------------------------------------------------------
# Build index
# ---------------------------------------------------------

print("Building blocking index...")

indexes = {
    "NAME": defaultdict(set),
    "NAMELAST": defaultdict(set),
    "NUM": defaultdict(set),
    "ADDR": defaultdict(set),
}

for row in sources.itertuples(index=False):

    row_dict = {
        "entity_id": row.entity_id,
        "business_name": row.business_name,
        "business_address": row.business_address,
        "country": row.country,
    }

    keys = create_blocking_keys(row_dict)

    for key in keys:

        parts = key.split("|")

        if len(parts) != 3:
            continue

        family = parts[1]

        if family in indexes:
            indexes[family][key].add(row.entity_id)

print("Raw indexes built.")

# ---------------------------------------------------------
# Frequency filtering
# ---------------------------------------------------------

for family in indexes:

    indexes[family] = {
        key: ids
        for key, ids in indexes[family].items()
        if len(ids) <= MAX_FREQUENCY
    }

    print(
        family,
        "usable keys:",
        len(indexes[family])
    )


# ---------------------------------------------------------
# Precompute S1 keys
# ---------------------------------------------------------

print("Preparing Source 1 keys...")

s1_key_map = {}

for row in s1.itertuples(index=False):

    row_dict = {
        "entity_id": row.entity_id,
        "business_name": row.business_name,
        "business_address": row.business_address,
        "country": row.country,
    }

    family_keys = {
        "NAME": [],
        "NAMELAST": [],
        "NUM": [],
        "ADDR": [],
    }

    for key in create_blocking_keys(row_dict):

        parts = key.split("|")

        if len(parts) != 3:
            continue

        family = parts[1]

        if family in family_keys:
            family_keys[family].append(key)

    s1_key_map[row.entity_id] = family_keys


# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------

def evaluate(families):

    total_true = 0
    captured = 0
    candidate_counts = []

    for s1_id, family_keys in s1_key_map.items():

        true_matches = ground_truth.get(
            s1_id,
            set()
        )

        total_true += len(true_matches)

        candidates = set()

        for family in families:

            for key in family_keys[family]:

                candidates.update(
                    indexes[family].get(
                        key,
                        set()
                    )
                )

        captured += len(
            true_matches & candidates
        )

        candidate_counts.append(
            len(candidates)
        )

    recall = (
        captured / total_true * 100
        if total_true
        else 0
    )

    candidate_counts.sort()

    avg = sum(candidate_counts) / len(candidate_counts)
    median = candidate_counts[len(candidate_counts) // 2]
    maximum = candidate_counts[-1]

    return recall, avg, median, maximum


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print()
print("=" * 75)
print("FAST BLOCKING FAMILY ANALYSIS")
print("=" * 75)

tests = [
    ["NAME"],
    ["NAMELAST"],
    ["NUM"],
    ["ADDR"],
    ["NAME", "NAMELAST"],
    ["NAME", "NUM"],
    ["NAME", "ADDR"],
    ["NAME", "NAMELAST", "NUM"],
    ["NAME", "NAMELAST", "NUM", "ADDR"],
]

for families in tests:

    recall, avg, median, maximum = evaluate(
        families
    )

    print(
        f"{'+'.join(families):30s}"
        f" Recall={recall:6.2f}%"
        f" Avg={avg:7.2f}"
        f" Median={median:4d}"
        f" Max={maximum:4d}"
    )

print("=" * 75)