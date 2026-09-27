import re


def normalize_for_blocking(value):
    """
    Lightweight normalization for blocking.

    Prefer the normalized columns produced by preprocessing.
    This fallback is only used if a normalized value is unavailable.
    """
    if value is None:
        return ""

    value = str(value).lower().strip()

    # Keep Unicode letters/numbers and whitespace.
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)

    return " ".join(value.split())


def get_blocking_value(row, normalized_column, raw_column):
    """
    Use Priyanka's normalized column when available.
    Fall back to the raw column.
    """
    value = row.get(normalized_column, "")

    if value is None or str(value).strip() == "":
        value = row.get(raw_column, "")

    return normalize_for_blocking(value)


def create_blocking_keys(row):
    """
    Create multiple blocking keys for one business record.

    Expected columns:
        entity_id
        business_name
        business_address
        country

    Preferred normalized columns:
        business_name_normalized
        business_address_normalized
    """

    country = str(row.get("country", "")).lower().strip()

    name = get_blocking_value(
        row,
        "business_name_normalized",
        "business_name",
    )

    address = get_blocking_value(
        row,
        "business_address_normalized",
        "business_address",
    )

    name_words = name.split()
    address_words = address.split()

    keys = set()

    # -------------------------------------------------
    # Name token blocking
    # -------------------------------------------------
    for word in name_words:
        if len(word) >= 4:
            keys.add(f"{country}|NAME|{word}")

    # -------------------------------------------------
    # Last name token blocking
    # -------------------------------------------------
    for word in reversed(name_words):
        if len(word) >= 4:
            keys.add(f"{country}|NAMELAST|{word}")
            break

    # -------------------------------------------------
    # Address number blocking
    # -------------------------------------------------
    for word in address_words:
        digits = re.sub(r"\D", "", word)

        if len(digits) >= 2:
            keys.add(f"{country}|NUM|{digits}")

    # -------------------------------------------------
    # Address token blocking
    # -------------------------------------------------
    for word in address_words:
        if len(word) >= 5:
            keys.add(f"{country}|ADDR|{word}")

    return keys