import regex


def normalize_business_address(address):
    """
    Safely normalize a business address.

    Rules:
    - Convert to lowercase
    - Trim leading/trailing spaces
    - Preserve Unicode letters, combining marks, and numbers
    - Replace punctuation/symbols with spaces
    - Collapse repeated spaces
    - Preserve the original address order
    """

    if address is None:
        return ""

    address = str(address).lower().strip()

    # Preserve Unicode letters, combining marks, numbers, and whitespace.
    address = regex.sub(r"[^\p{L}\p{M}\p{N}\s]", " ", address)

    # Collapse multiple spaces
    address = regex.sub(r"\s+", " ", address).strip()

    return address