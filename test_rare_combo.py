import pandas as pd
import re
from collections import defaultdict, Counter


N = 10_000
MAX_FREQUENCY = 100


def normalize(value):
    value = str(value).lower()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def tokens(value, min_len):
    return {
        word
        for word in normalize(value).split()
        if len(word) >= min_len
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
# STEP 1: Calculate global token frequencies
# ---------------------------------------------------------

print("Calculating token frequencies...")

name_frequency = Counter()
address_frequency = Counter()

for row in sources.itertuples(index=False):

    name_frequency.update(
        tokens(row.business_name, 4)
    )

    address_frequency.update(
        tokens(row.business_address, 5)
    )


print(
    "Name tokens:",
    len(name_frequency)
)

print(
    "Address tokens:",
    len(address_frequency)
)


# ---------------------------------------------------------
# STEP 2: Build rare-token combination index
# ---------------------------------------------------------

print("Building rare name + address index...")

combo_index = defaultdict(set)

for row in sources.itertuples(index=False):

    name_tokens = [
        token
        for token in tokens(row.business_name, 4)
        if name_frequency[token] <= MAX_FREQUENCY
    ]

    address_tokens = [
        token
        for token in tokens(row.business_address, 5)
        if address_frequency[token] <= MAX_FREQUENCY
    ]

    for name_token in name_tokens:

        for address_token in address_tokens:

            key = (
                f"{str(row.country).lower()}|"
                f"{name_token}|"
                f"{address_token}"
            )

            combo_index[key].add(
                row.entity_id
            )


print(
    "Raw combination keys:",
    len(combo_index)
)


# ---------------------------------------------------------
# STEP 3: Remove overly common combinations
# ---------------------------------------------------------

combo_index = {
    key: ids
    for key, ids in combo_index.items()
    if len(ids) <= MAX_FREQUENCY
}


print(
    "Usable combination keys:",
    len(combo_index)
)


# ---------------------------------------------------------
# STEP 4: Evaluate
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

    name_tokens = [
        token
        for token in tokens(row.business_name, 4)
        if name_frequency[token] <= MAX_FREQUENCY
    ]

    address_tokens = [
        token
        for token in tokens(row.business_address, 5)
        if address_frequency[token] <= MAX_FREQUENCY
    ]

    candidates = set()

    for name_token in name_tokens:

        for address_token in address_tokens:

            key = (
                f"{str(row.country).lower()}|"
                f"{name_token}|"
                f"{address_token}"
            )

            candidates.update(
                combo_index.get(
                    key,
                    set()
                )
            )

    captured_true += len(
        true_matches & candidates
    )

    candidate_counts.append(
        len(candidates)
    )


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

candidate_counts.sort()

recall = (
    captured_true / total_true * 100
    if total_true
    else 0
)

print()
print("=" * 65)
print("RARE NAME + ADDRESS COMBINATION RESULTS")
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