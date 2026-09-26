import pandas as pd
import re
from collections import defaultdict


N = 10_000
MAX_FREQUENCY = 100


def normalize_name(value):
    value = str(value).lower()
    value = re.sub(r"[^a-z0-9]", "", value)
    return value


def char_trigrams(value):
    value = normalize_name(value)

    if len(value) < 3:
        return set()

    return {
        value[i:i+3]
        for i in range(len(value) - 2)
    }


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


# ---------------------------------------------------------
# Build character trigram index
# ---------------------------------------------------------

print("Building character trigram index...")

trigram_index = defaultdict(set)

for row in sources.itertuples(index=False):

    trigrams = char_trigrams(row.business_name)

    for trigram in trigrams:
        trigram_index[trigram].add(row.entity_id)


print("Raw trigrams:", len(trigram_index))


# ---------------------------------------------------------
# Remove common trigrams
# ---------------------------------------------------------

trigram_index = {
    trigram: ids
    for trigram, ids in trigram_index.items()
    if len(ids) <= MAX_FREQUENCY
}

print(
    "Usable trigrams:",
    len(trigram_index)
)


# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------

total_true = 0
captured_true = 0

candidate_counts = []


for row in s1.itertuples(index=False):

    s1_id = row.entity_id

    true_matches = ground_truth.get(
        s1_id,
        set()
    )

    total_true += len(true_matches)

    candidates = set()

    for trigram in char_trigrams(row.business_name):

        candidates.update(
            trigram_index.get(
                trigram,
                set()
            )
        )

    captured_true += len(
        true_matches & candidates
    )

    candidate_counts.append(
        len(candidates)
    )


recall = (
    captured_true / total_true * 100
    if total_true
    else 0
)

candidate_counts.sort()

print()
print("=" * 60)
print("CHARACTER BLOCKING RESULTS")
print("=" * 60)

print("Total true links:", total_true)

print("Captured true links:", captured_true)

print(
    "Character blocking recall:",
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

print("=" * 60)