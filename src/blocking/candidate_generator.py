import re


def normalize_text(value):
    """Convert text into simple alphanumeric tokens."""
    value = str(value).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return value.split()


def create_blocking_keys(row):
    """
    Create multiple blocking keys for one business record.

    Expected columns:
        entity_id
        business_name
        business_address
        country
    """

    country = str(row.get("country", "")).lower().strip()

    name_words = normalize_text(
        row.get("business_name", "")
    )

    address_words = normalize_text(
        row.get("business_address", "")
    )

    keys = set()

    # -----------------------------------------
    # Address-number blocking
    # -----------------------------------------

    for word in address_words:

        if any(char.isdigit() for char in word):

            number = re.sub(
                r"[^0-9]",
                "",
                word
            )

            if len(number) >= 2:

                keys.add(
                    f"{country}|NUM|{number}"
                )

                break

    # -----------------------------------------
    # Address-location blocking
    # -----------------------------------------

    for word in address_words[-4:]:

        if len(word) >= 4:

            keys.add(
                f"{country}|LOC|{word}"
            )

    # -----------------------------------------
    # Business-name blocking
    # -----------------------------------------

    for word in name_words:

        if len(word) >= 4:

            keys.add(
                f"{country}|NAME|{word}"
            )

            break

    # -----------------------------------------
    # Last business-name token
    # -----------------------------------------

    for word in reversed(name_words):

        if len(word) >= 4:

            keys.add(
                f"{country}|NAMELAST|{word}"
            )

            break

    return keys


def build_blocking_index(source1):
    """
    Build an inverted index for Source 1.

    Returns:
        dictionary:
            blocking_key -> set of Source1 entity IDs
    """

    blocking_index = {}

    for _, row in source1.iterrows():

        entity_id = row["entity_id"]

        keys = create_blocking_keys(row)

        for key in keys:

            if key not in blocking_index:
                blocking_index[key] = set()

            blocking_index[key].add(entity_id)

    return blocking_index


def generate_candidates(source_df, blocking_index):
    """
    Generate candidate Source1 IDs for Source2/Source3 records.

    Returns:
        list of tuples:
            (source1_entity_id, source_entity_id)
    """

    candidates = []

    for _, row in source_df.iterrows():

        source_entity_id = row["entity_id"]

        keys = create_blocking_keys(row)

        candidate_ids = set()

        for key in keys:

            candidate_ids.update(
                blocking_index.get(key, set())
            )

        for source1_id in candidate_ids:

            candidates.append(
                (
                    source1_id,
                    source_entity_id
                )
            )

    return candidates