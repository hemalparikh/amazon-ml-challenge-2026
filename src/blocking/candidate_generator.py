import argparse
import os
import re
from collections import defaultdict

import pandas as pd


DEFAULT_NORMAL_CAP = 100
DEFAULT_STRONG_CAP = 25
DEFAULT_CHUNK_SIZE = 100_000


# ============================================================
# TEXT HELPERS
# ============================================================

def tokenize(value):
    """Convert text into lowercase alphanumeric tokens."""

    if pd.isna(value):
        return []

    value = str(value).lower()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value
    )

    return value.split()


def meaningful_tokens(tokens):
    """Keep meaningful address/name tokens."""

    return [
        token
        for token in tokens
        if len(token) >= 3
    ]


def get_address_number(address_tokens):
    """Return the first useful numeric address component."""

    for token in address_tokens:

        digits = re.sub(
            r"[^0-9]",
            "",
            token
        )

        if len(digits) >= 2:
            return digits

    return None


# ============================================================
# NORMAL BLOCKING
# ============================================================

def create_normal_keys(row):
    """
    Existing normal blocking rules.

    Keys:
        ADDR
        NAME
        NAMELAST
        NUM
    """

    country = str(
        row.get("country", "")
    ).lower().strip()

    name_tokens = meaningful_tokens(
        tokenize(
            row.get("business_name", "")
        )
    )

    address_tokens = meaningful_tokens(
        tokenize(
            row.get("business_address", "")
        )
    )

    keys = set()

    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    if name_tokens:

        keys.add(
            f"NAME|{country}|{name_tokens[0]}"
        )

    # --------------------------------------------------------
    # NAMELAST
    # --------------------------------------------------------

    if len(name_tokens) >= 2:

        keys.add(
            f"NAMELAST|{country}|{name_tokens[-1]}"
        )

    # --------------------------------------------------------
    # ADDR
    # --------------------------------------------------------

    for token in address_tokens[-4:]:

        if len(token) >= 3:

            keys.add(
                f"ADDR|{country}|{token}"
            )

    # --------------------------------------------------------
    # NUM
    # --------------------------------------------------------

    number = get_address_number(
        address_tokens
    )

    if number is not None:

        keys.add(
            f"NUM|{country}|{number}"
        )

    return keys


# ============================================================
# STRONG ADDRESS BLOCKING
# ============================================================

def create_strong_address_keys(row):
    """
    Strong address blocker:

        country + address number + 2 meaningful
        address tokens

    All unique token pairs are considered.
    """

    country = str(
        row.get("country", "")
    ).lower().strip()

    address_tokens = meaningful_tokens(
        tokenize(
            row.get("business_address", "")
        )
    )

    number = get_address_number(
        address_tokens
    )

    if number is None:
        return set()

    # Remove the token containing the address number.
    non_number_tokens = []

    for token in address_tokens:

        digits = re.sub(
            r"[^0-9]",
            "",
            token
        )

        if digits == number:
            continue

        non_number_tokens.append(token)

    # Remove duplicates while preserving order.
    unique_tokens = list(
        dict.fromkeys(
            non_number_tokens
        )
    )

    if len(unique_tokens) < 2:
        return set()

    keys = set()

    # Generate combinations of two meaningful
    # address tokens.
    for i in range(
        len(unique_tokens)
    ):

        for j in range(
            i + 1,
            len(unique_tokens)
        ):

            token1 = unique_tokens[i]
            token2 = unique_tokens[j]

            keys.add(
                f"STRONG|{country}|{number}|"
                f"{token1}|{token2}"
            )

    return keys


# ============================================================
# INDEX BUILDING
# ============================================================

def build_index(
    source_paths,
    normal_cap=DEFAULT_NORMAL_CAP,
    strong_cap=DEFAULT_STRONG_CAP,
    chunk_size=DEFAULT_CHUNK_SIZE,
):
    """
    Build separate normal and strong blocking indexes.

    Normal keys use normal_cap.
    Strong address keys use strong_cap.
    """

    normal_index = defaultdict(list)
    strong_index = defaultdict(list)

    normal_frequency = defaultdict(int)
    strong_frequency = defaultdict(int)

    for source_path in source_paths:

        print(
            f"Indexing {os.path.basename(source_path)}..."
        )

        for chunk in pd.read_csv(
            source_path,
            sep="\t",
            chunksize=chunk_size,
            dtype=str,
        ):

            for _, row in chunk.iterrows():

                entity_id = row["entity_id"]

                # --------------------------------------------
                # Normal blocker
                # --------------------------------------------

                normal_keys = create_normal_keys(
                    row
                )

                for key in normal_keys:

                    normal_frequency[key] += 1

                    if (
                        normal_frequency[key]
                        <= normal_cap
                    ):
                        normal_index[key].append(
                            entity_id
                        )

                # --------------------------------------------
                # Strong address blocker
                # --------------------------------------------

                strong_keys = create_strong_address_keys(
                    row
                )

                for key in strong_keys:

                    strong_frequency[key] += 1

                    if (
                        strong_frequency[key]
                        <= strong_cap
                    ):
                        strong_index[key].append(
                            entity_id
                        )

    # Remove keys that exceeded their caps.
    normal_index = {
        key: ids
        for key, ids in normal_index.items()
        if normal_frequency[key] <= normal_cap
    }

    strong_index = {
        key: ids
        for key, ids in strong_index.items()
        if strong_frequency[key] <= strong_cap
    }

    print(
        f"Normal keys retained: "
        f"{len(normal_index):,}"
    )

    print(
        f"Strong address keys retained: "
        f"{len(strong_index):,}"
    )

    return normal_index, strong_index


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidates(
    source1_path,
    normal_index,
    strong_index,
    output_path,
    chunk_size=DEFAULT_CHUNK_SIZE,
):
    """
    Generate Source1 -> Source2/Source3 candidates.

    IMPORTANT:
    candidate_ids is a SET so candidates reached through
    multiple blocking keys are counted only once.
    """

    source1_count = 0
    total_candidate_links = 0

    with open(
        output_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as output:

        output.write(
            "source1_entity_id\tcandidate_entity_ids\n"
        )

        for chunk in pd.read_csv(
            source1_path,
            sep="\t",
            chunksize=chunk_size,
            dtype=str,
        ):

            for _, row in chunk.iterrows():

                source1_id = row["entity_id"]

                # IMPORTANT:
                # Use a SET for the union of all
                # blocking-key results.
                candidate_ids = set()

                # ------------------------------------------------
                # Normal blocking candidates
                # ------------------------------------------------

                normal_keys = create_normal_keys(
                    row
                )

                for key in normal_keys:

                    candidate_ids.update(
                        normal_index.get(
                            key,
                            []
                        )
                    )

                # ------------------------------------------------
                # Strong address candidates
                # ------------------------------------------------

                strong_keys = create_strong_address_keys(
                    row
                )

                for key in strong_keys:

                    candidate_ids.update(
                        strong_index.get(
                            key,
                            []
                        )
                    )

                # ------------------------------------------------
                # Final candidate count
                # ------------------------------------------------

                # This is intentionally AFTER the union.
                candidate_count = len(
                    candidate_ids
                )

                candidate_ids = {
                    candidate_id
                    for candidate_id in candidate_ids
                    if (
                        candidate_id.startswith("S2-")
                        or candidate_id.startswith("S3-")
                    )
                }

                sorted_candidates = sorted(
                    candidate_ids
                )

                output.write(
                    source1_id
                    + "\t"
                    + ",".join(
                        sorted_candidates
                    )
                    + "\n"
                )

                source1_count += 1

                # Count unique candidates for this S1.
                total_candidate_links += len(
                    candidate_ids
                )

    return (
        source1_count,
        total_candidate_links
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate Source1 candidate IDs "
            "using normal + strong address blocking."
        )
    )

    parser.add_argument(
        "--test-dir",
        default=(
            r"C:\Users\bhama\Downloads"
            r"\6ab10eb3b23ba_student_resource"
            r"\student_resource\dataset\test"
        ),
    )

    parser.add_argument(
        "--output",
        default="candidate_pairs.tsv",
    )

    parser.add_argument(
        "--normal-cap",
        type=int,
        default=DEFAULT_NORMAL_CAP,
    )

    parser.add_argument(
        "--strong-cap",
        type=int,
        default=DEFAULT_STRONG_CAP,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
    )

    args = parser.parse_args()

    test_dir = args.test_dir

    source1 = os.path.join(
        test_dir,
        "test_source1.tsv"
    )

    source2 = os.path.join(
        test_dir,
        "test_source2.tsv"
    )

    source3 = os.path.join(
        test_dir,
        "test_source3.tsv"
    )

    print("=" * 60)
    print("BLOCKING / CANDIDATE GENERATION")
    print("=" * 60)

    print(
        f"Normal frequency cap: "
        f"{args.normal_cap}"
    )

    print(
        f"Strong address frequency cap: "
        f"{args.strong_cap}"
    )

    print(
        f"Chunk size: "
        f"{args.chunk_size:,}"
    )

    print()

    # Build Source2 + Source3 indexes.
    normal_index, strong_index = build_index(
        source_paths=[
            source2,
            source3,
        ],
        normal_cap=args.normal_cap,
        strong_cap=args.strong_cap,
        chunk_size=args.chunk_size,
    )

    print()
    print("Generating Source1 candidates...")

    source1_count, candidate_count = (
        generate_candidates(
            source1_path=source1,
            normal_index=normal_index,
            strong_index=strong_index,
            output_path=args.output,
            chunk_size=args.chunk_size,
        )
    )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"Source1 entities: "
        f"{source1_count:,}"
    )

    print(
        f"Candidate links: "
        f"{candidate_count:,}"
    )

    if source1_count:

        print(
            "Average candidates/S1: "
            f"{candidate_count / source1_count:.2f}"
        )

    print(
        f"Output: {args.output}"
    )


if __name__ == "__main__":
    main()