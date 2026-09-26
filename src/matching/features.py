import re

from rapidfuzz.fuzz import (
    ratio,
    token_set_ratio,
    token_sort_ratio,
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_business_name(name):
    """
    Normalize a business name.

    Handles:
    - missing values
    - lowercase conversion
    - punctuation removal
    - common legal suffixes
    - whitespace normalization
    """

    if name is None:
        return ""

    name = str(name).strip()

    if not name or name.lower() == "nan":
        return ""

    name = name.lower()

    # Replace punctuation/symbols with spaces.
    name = re.sub(r"[^\w\s]", " ", name, flags=re.UNICODE)

    # Standardize common business/legal abbreviations.
    replacements = {
        r"\bpvt\b": "private",
        r"\bpriv\b": "private",
        r"\bltd\b": "limited",
        r"\blt\b": "limited",
        r"\bcorp\b": "corporation",
        r"\binc\b": "incorporated",
        r"\bco\b": "company",
        r"\bllc\b": "llc",
        r"\bplc\b": "plc",
    }

    for pattern, replacement in replacements.items():
        name = re.sub(pattern, replacement, name)

    # Collapse whitespace.
    name = re.sub(r"\s+", " ", name).strip()

    return name


def normalize_business_address(address):
    """
    Normalize a business address.

    Handles:
    - missing values
    - lowercase conversion
    - punctuation removal
    - common address abbreviations
    - whitespace normalization
    """

    if address is None:
        return ""

    address = str(address).strip()

    if not address or address.lower() == "nan":
        return ""

    address = address.lower()

    # Replace punctuation/symbols with spaces.
    address = re.sub(r"[^\w\s]", " ", address, flags=re.UNICODE)

    # Common address abbreviations.
    abbreviation_rules = {
        r"\brd\b": "road",
        r"\bst\b": "street",
        r"\bave\b": "avenue",
        r"\bav\b": "avenue",
        r"\bdr\b": "drive",
        r"\bblvd\b": "boulevard",
        r"\bhwy\b": "highway",
        r"\bln\b": "lane",
        r"\bpkwy\b": "parkway",
        r"\bct\b": "court",
        r"\bpl\b": "place",
        r"\bapt\b": "apartment",
        r"\bste\b": "suite",
    }

    for pattern, replacement in abbreviation_rules.items():
        address = re.sub(pattern, replacement, address)

    address = re.sub(r"\s+", " ", address).strip()

    return address


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def first_token(text):
    """Return the first token."""

    if not text:
        return ""

    return text.split()[0]


def tokens(text):
    """Return whitespace-separated tokens."""

    if not text:
        return set()

    return set(text.split())


def important_tokens(text):
    """
    Return business-name tokens excluding common legal suffixes.
    """

    legal_tokens = {
        "private",
        "limited",
        "corporation",
        "incorporated",
        "company",
        "llc",
        "plc",
    }

    return {
        token
        for token in tokens(text)
        if token not in legal_tokens
    }


def numeric_tokens(text):
    """
    Extract numeric sequences.
    """

    if not text:
        return set()

    return set(re.findall(r"\d+", text))


def postal_tokens(text):
    """
    Extract possible postal/ZIP codes.

    Kept generic because the dataset contains multiple countries.
    """

    if not text:
        return set()

    numbers = numeric_tokens(text)

    return {
        number
        for number in numbers
        if 4 <= len(number) <= 6
    }


def token_overlap(text1, text2):
    """
    Intersection divided by the smaller token set.
    """

    tokens1 = tokens(text1)
    tokens2 = tokens(text2)

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1.intersection(tokens2)

    return len(intersection) / min(len(tokens1), len(tokens2))


def important_token_overlap(text1, text2):
    """Calculate overlap between important business-name tokens."""

    tokens1 = important_tokens(text1)
    tokens2 = important_tokens(text2)

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1.intersection(tokens2)

    return len(intersection) / min(len(tokens1), len(tokens2))


def safe_ratio(value1, value2):
    """Calculate fuzzy similarity safely."""

    if not value1 or not value2:
        return 0.0

    return ratio(value1, value2) / 100.0


def safe_token_set_ratio(value1, value2):
    """Calculate token-set similarity safely."""

    if not value1 or not value2:
        return 0.0

    return token_set_ratio(value1, value2) / 100.0


def safe_token_sort_ratio(value1, value2):
    """Calculate token-sort similarity safely."""

    if not value1 or not value2:
        return 0.0

    return token_sort_ratio(value1, value2) / 100.0


# ============================================================
# FEATURE CREATION
# ============================================================

def create_features(df):
    """
    Create entity-matching features for candidate pairs.

    Expected raw columns:

        source1_name
        candidate_name
        source1_address
        candidate_address
        source1_country
        candidate_country

    Optional normalized columns:

        source1_name_normalized
        candidate_name_normalized
        source1_address_normalized
        candidate_address_normalized

    If normalized columns already exist, they are used directly.
    Otherwise, fallback normalization is performed.
    """

    df = df.copy()

    # ========================================================
    # NAME NORMALIZATION
    # ========================================================

    # Use project preprocessing output if already available.
    if "source1_name_normalized" not in df.columns:
        df["source1_name_normalized"] = (
            df["source1_name"]
            .fillna("")
            .apply(normalize_business_name)
        )

    if "candidate_name_normalized" not in df.columns:
        df["candidate_name_normalized"] = (
            df["candidate_name"]
            .fillna("")
            .apply(normalize_business_name)
        )

    # ========================================================
    # NAME FEATURES
    # ========================================================

    # Exact normalized name.
    df["name_normalized_exact_match"] = (
        df["source1_name_normalized"]
        == df["candidate_name_normalized"]
    ).astype(int)

    # Character-level similarity.
    df["name_similarity"] = df.apply(
        lambda row: safe_ratio(
            row["source1_name_normalized"],
            row["candidate_name_normalized"],
        ),
        axis=1,
    )

    # Token-set similarity.
    df["name_token_set_similarity"] = df.apply(
        lambda row: safe_token_set_ratio(
            row["source1_name_normalized"],
            row["candidate_name_normalized"],
        ),
        axis=1,
    )

    # Token-sort similarity.
    df["name_token_sort_similarity"] = df.apply(
        lambda row: safe_token_sort_ratio(
            row["source1_name_normalized"],
            row["candidate_name_normalized"],
        ),
        axis=1,
    )

    # First token match.
    df["name_first_token_match"] = (
        df["source1_name_normalized"].apply(first_token)
        == df["candidate_name_normalized"].apply(first_token)
    ).astype(int)

    # Important-token overlap.
    df["name_important_token_overlap"] = df.apply(
        lambda row: important_token_overlap(
            row["source1_name_normalized"],
            row["candidate_name_normalized"],
        ),
        axis=1,
    )

    # Similarity after ignoring legal suffixes.
    df["name_important_token_similarity"] = df.apply(
        lambda row: safe_token_set_ratio(
            " ".join(
                sorted(
                    important_tokens(
                        row["source1_name_normalized"]
                    )
                )
            ),
            " ".join(
                sorted(
                    important_tokens(
                        row["candidate_name_normalized"]
                    )
                )
            ),
        ),
        axis=1,
    )

    # Number of name tokens.
    df["source1_name_token_count"] = (
        df["source1_name_normalized"]
        .apply(lambda x: len(tokens(x)))
    )

    df["candidate_name_token_count"] = (
        df["candidate_name_normalized"]
        .apply(lambda x: len(tokens(x)))
    )

    # Missing name indicators.
    df["source1_name_missing"] = (
        df["source1_name_normalized"] == ""
    ).astype(int)

    df["candidate_name_missing"] = (
        df["candidate_name_normalized"] == ""
    ).astype(int)

    # ========================================================
    # ADDRESS NORMALIZATION
    # ========================================================

    # Use project preprocessing output if already available.
    if "source1_address_normalized" not in df.columns:
        df["source1_address_normalized"] = (
            df["source1_address"]
            .fillna("")
            .apply(normalize_business_address)
        )

    if "candidate_address_normalized" not in df.columns:
        df["candidate_address_normalized"] = (
            df["candidate_address"]
            .fillna("")
            .apply(normalize_business_address)
        )

    # ========================================================
    # ADDRESS FEATURES
    # ========================================================

    # Exact normalized address.
    df["address_exact_match"] = (
        df["source1_address_normalized"]
        == df["candidate_address_normalized"]
    ).astype(int)

    # Character-level similarity.
    df["address_similarity"] = df.apply(
        lambda row: safe_ratio(
            row["source1_address_normalized"],
            row["candidate_address_normalized"],
        ),
        axis=1,
    )

    # Token-set similarity.
    df["address_token_set_similarity"] = df.apply(
        lambda row: safe_token_set_ratio(
            row["source1_address_normalized"],
            row["candidate_address_normalized"],
        ),
        axis=1,
    )

    # Token-sort similarity.
    df["address_token_sort_similarity"] = df.apply(
        lambda row: safe_token_sort_ratio(
            row["source1_address_normalized"],
            row["candidate_address_normalized"],
        ),
        axis=1,
    )

    # Token overlap.
    df["address_token_overlap"] = df.apply(
        lambda row: token_overlap(
            row["source1_address_normalized"],
            row["candidate_address_normalized"],
        ),
        axis=1,
    )

    # Any common numeric token.
    df["address_number_match"] = df.apply(
        lambda row: int(
            bool(
                numeric_tokens(
                    row["source1_address_normalized"]
                )
                &
                numeric_tokens(
                    row["candidate_address_normalized"]
                )
            )
        ),
        axis=1,
    )

    # Postal-code match.
    df["address_postal_code_match"] = df.apply(
        lambda row: int(
            bool(
                postal_tokens(
                    row["source1_address_normalized"]
                )
                &
                postal_tokens(
                    row["candidate_address_normalized"]
                )
            )
        ),
        axis=1,
    )

    # Address token counts.
    df["source1_address_token_count"] = (
        df["source1_address_normalized"]
        .apply(lambda x: len(tokens(x)))
    )

    df["candidate_address_token_count"] = (
        df["candidate_address_normalized"]
        .apply(lambda x: len(tokens(x)))
    )

    # Missing address indicators.
    df["source1_address_missing"] = (
        df["source1_address_normalized"] == ""
    ).astype(int)

    df["candidate_address_missing"] = (
        df["candidate_address_normalized"] == ""
    ).astype(int)

    # ========================================================
    # COUNTRY FEATURES
    # ========================================================

    df["source1_country_normalized"] = (
        df["source1_country"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["candidate_country_normalized"] = (
        df["candidate_country"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Handle string "nan".
    df.loc[
        df["source1_country_normalized"] == "nan",
        "source1_country_normalized",
    ] = ""

    df.loc[
        df["candidate_country_normalized"] == "nan",
        "candidate_country_normalized",
    ] = ""

    # Exact country match.
    df["country_exact_match"] = (
        df["source1_country_normalized"]
        == df["candidate_country_normalized"]
    ).astype(int)

    # Country missing indicators.
    df["source1_country_missing"] = (
        df["source1_country_normalized"] == ""
    ).astype(int)

    df["candidate_country_missing"] = (
        df["candidate_country_normalized"] == ""
    ).astype(int)

    return df