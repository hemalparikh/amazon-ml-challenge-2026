import regex


def normalize_business_name(name):
    """
    Safely normalize a business name.

    Rules:
    - Convert to lowercase
    - Trim leading/trailing spaces
    - Preserve Unicode letters, combining marks, and numbers
    - Replace punctuation/symbols with spaces
    - Collapse repeated spaces
    """

    if name is None:
        return ""

    name = str(name).lower().strip()

    # Preserve Unicode letters, combining marks, numbers, and whitespace.
    name = regex.sub(r"[^\p{L}\p{M}\p{N}\s]", " ", name)

    # Collapse multiple spaces
    name = regex.sub(r"\s+", " ", name).strip()

    return name