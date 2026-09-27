import pandas as pd

from preprocessing.name_normalizer import normalize_business_name
from preprocessing.address_normalizer import normalize_business_address


def preprocess_data(df):
    """
    Apply safe normalization to business names and addresses.

    Original columns are preserved.
    New normalized columns are added.
    """

    df = df.copy()

    df["business_name_normalized"] = df["business_name"].map(
        lambda value: "" if pd.isna(value) else normalize_business_name(value)
    )

    df["business_address_normalized"] = df["business_address"].map(
        normalize_business_address
    )

    return df
