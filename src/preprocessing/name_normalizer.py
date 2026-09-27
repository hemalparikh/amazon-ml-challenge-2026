import regex


def normalize_business_name(name):
    """
    Safely normalize a business name.

    Rules:
    - Convert to lowercase
    - Trim leading/trailing spaces
    - Preserve Unicode letters, combining marks, and numbers
    - Replace punctuation/symbols with spaces
    - Standardize common legal business terms
    - Collapse repeated spaces
    """

    if name is None:
        return ""

    name = str(name).lower().strip()

    # Preserve Unicode letters, combining marks, numbers, and whitespace.
    name = regex.sub(r"[^\p{L}\p{M}\p{N}\s]", " ", name)

    # Standardize common legal business terms.
    name = regex.sub(
        r"\bprivate\s+limited\b",
        "pvt ltd",
        name
    )

    name = regex.sub(
        r"\bpvt\s+(?:limited|ltd)\b",
        "pvt ltd",
        name
    )

    name = regex.sub(
        r"\blimited\b",
        "ltd",
        name
    )

    name = regex.sub(
        r"\bcorporation\b",
        "corp",
        name
    )

    name = regex.sub(
        r"\bcompany\b",
        "co",
        name
    )

    # Collapse multiple spaces.
    name = regex.sub(r"\s+", " ", name).strip()

    return name