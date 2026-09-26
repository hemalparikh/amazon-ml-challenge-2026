from rapidfuzz.fuzz import ratio, token_set_ratio


def normalize_name(name):
    return " ".join(str(name).lower().split())


def normalize_business_name(name):
    """
    Normalize common business-name abbreviations.
    """
    name = normalize_name(name)

    replacements = {
        "private limited": "pvt ltd",
        "private": "pvt",
        "limited": "ltd",
        "corporation": "corp",
        "company": "co"
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    return " ".join(name.split())


def create_features(df):
    df = df.copy()

    # --------------------------------
    # Normalize names
    # --------------------------------

    df["source1_name_normalized"] = (
        df["source1_name"].apply(normalize_business_name)
    )

    df["candidate_name_normalized"] = (
        df["candidate_name"].apply(normalize_business_name)
    )

    # --------------------------------
    # Exact name match
    # --------------------------------

    df["name_normalized_exact_match"] = (
        df["source1_name_normalized"]
        == df["candidate_name_normalized"]
    ).astype(int)

    # --------------------------------
    # Overall name similarity
    # --------------------------------

    df["name_similarity"] = df.apply(
        lambda row: ratio(
            row["source1_name_normalized"],
            row["candidate_name_normalized"]
        ) / 100,
        axis=1
    )

    # --------------------------------
    # Token-based name similarity
    # --------------------------------

    df["name_token_similarity"] = df.apply(
        lambda row: token_set_ratio(
            row["source1_name_normalized"],
            row["candidate_name_normalized"]
        ) / 100,
        axis=1
    )

    # --------------------------------
    # First-word match
    # --------------------------------

    df["name_first_token_match"] = df.apply(
        lambda row: (
            row["source1_name_normalized"].split()[0]
            == row["candidate_name_normalized"].split()[0]
        )
        if row["source1_name_normalized"]
        and row["candidate_name_normalized"]
        else 0,
        axis=1
    ).astype(int)

    # --------------------------------
    # Address exact match
    # --------------------------------

    df["address_exact_match"] = (
        df["source1_address"].fillna("").str.strip().str.lower()
        == df["candidate_address"].fillna("").str.strip().str.lower()
    ).astype(int)

    # --------------------------------
    # Country exact match
    # --------------------------------

    df["country_exact_match"] = (
        df["source1_country"].fillna("").str.strip().str.lower()
        == df["candidate_country"].fillna("").str.strip().str.lower()
    ).astype(int)

    return df