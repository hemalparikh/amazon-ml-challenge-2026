import pandas as pd
from collections import defaultdict

from src.blocking.candidate_generator import create_blocking_keys
from src.preprocessing.preprocess import preprocess_data


N = 10_000
MAX_FREQUENCY = 100


print("Loading Source 1...")

s1_raw = pd.read_csv(
    "data/train/train_source1.tsv",
    sep="\t",
    dtype=str,
    nrows=N,
    keep_default_na=False
)

print("Preprocessing Source 1...")

s1 = preprocess_data(s1_raw)

s1_ids = set(s1["entity_id"])


print("Loading ground truth...")

gt = pd.read_csv(
    "data/train/train_ground_truth.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False
)

gt = gt[
    gt["source1_entity_id"].isin(s1_ids)
]

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


print("Preprocessing Source 2 + Source 3...")

sources = preprocess_data(sources)


# ---------------------------------------------------------
# Build normalized blocking index
# ---------------------------------------------------------

print("Building normalized blocking index...")

blocking_index = defaultdict(set)

for row in sources.itertuples(index=False):

    row_dict = {
        "entity_id": row.entity_id,
        "business_name": row.business_name,
        "business_address": row.business_address,
        "business_name_normalized": row.business_name_normalized,
        "business_address_normalized": row.business_address_normalized,
        "country": row.country,
    }

    for key in create_blocking_keys(row_dict):
        blocking_index[key].add(row.entity_id)


print(
    "Raw blocking keys:",
    len(blocking_index)
)


# ---------------------------------------------------------
# Remove very common keys
# ---------------------------------------------------------

blocking_index = {
    key: ids
    for key, ids in blocking_index.items()
    if len(ids) <= MAX_FREQUENCY
}


print(
    "Usable blocking keys:",
    len(blocking_index)
)


# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------

total_true = 0
captured_true = 0

candidate_counts = []


for row in s1.itertuples(index=False):

    true_matches = ground_truth.get(
        row.entity_id,
        set()
    )

    total_true += len(true_matches)

    row_dict = {
        "entity_id": row.entity_id,
        "business_name": row.business_name,
        "business_address": row.business_address,
        "business_name_normalized": row.business_name_normalized,
        "business_address_normalized": row.business_address_normalized,
        "country": row.country,
    }

    candidates = set()

    for key in create_blocking_keys(row_dict):
        candidates.update(
            blocking_index.get(key, set())
        )

    captured_true += len(
        true_matches & candidates
    )

    candidate_counts.append(
        len(candidates)
    )


candidate_counts.sort()

recall = (
    captured_true / total_true * 100
    if total_true
    else 0
)


print()
print("=" * 65)
print("NORMALIZED BLOCKING RESULTS")
print("=" * 65)

print(
    "Total true links:",
    total_true
)

print(
    "Captured true links:",
    captured_true
)

print(
    "Blocking recall:",
    round(recall, 2),
    "%"
)

print(
    "Average candidates:",
    round(
        sum(candidate_counts) /
        len(candidate_counts),
        2
    )
)

print(
    "Median candidates:",
    candidate_counts[
        len(candidate_counts) // 2
    ]
)

print(
    "Maximum candidates:",
    candidate_counts[-1]
)

print("=" * 65)