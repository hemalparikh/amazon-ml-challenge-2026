import regex


def normalize_business_address(address):
    """
    Safely normalize a business address.

    Rules:
    - Convert to lowercase
    - Trim leading/trailing spaces
    - Preserve Unicode letters, combining marks, and numbers
    - Replace punctuation/symbols with spaces
    - Standardize common address abbreviations
    - Collapse repeated spaces
    - Preserve the original address order
    """

    if address is None:
        return ""

    if regex.match(r"^\s*nan\s*$", str(address), regex.IGNORECASE):
        return ""

    address = str(address).lower().strip()

    # Preserve Unicode letters, combining marks, numbers, and whitespace.
    address = regex.sub(r"[^\p{L}\p{M}\p{N}\s]", " ", address)

    # Standardize common address abbreviations.
    abbreviation_rules = {
        r"\brd\b": "road",
        r"\bst\b": "street",
        r"\bave\b": "avenue",
        r"\bdr\b": "drive",
        r"\bblvd\b": "boulevard",
        r"\bhwy\b": "highway",
        r"\bln\b": "lane",
        r"\bpkwy\b": "parkway",
        r"\bct\b": "court",
        r"\bpl\b": "place",
    }

    for pattern, replacement in abbreviation_rules.items():
        address = regex.sub(pattern, replacement, address)

    # Collapse multiple spaces.
    address = regex.sub(r"\s+", " ", address).strip()

    return address

