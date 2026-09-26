import pandas as pd
from rapidfuzz.fuzz import ratio


# Load candidate pairs
file_path = "data/sample_candidates.csv"
df = pd.read_csv(file_path)


# Normalize business names
def normalize_name(name):
    return " ".join(name.lower().split())


df["source1_name_normalized"] = df["source1_name"].apply(normalize_name)
df["candidate_name_normalized"] = df["candidate_name"].apply(normalize_name)


# Normalized name exact match
df["name_normalized_exact_match"] = (
    df["source1_name_normalized"]
    == df["candidate_name_normalized"]
).astype(int)


# Name similarity
df["name_similarity"] = df.apply(
    lambda row: ratio(
        row["source1_name_normalized"],
        row["candidate_name_normalized"]
    ) / 100,
    axis=1
)


# Address exact match
df["address_exact_match"] = (
    df["source1_address"].str.strip().str.lower()
    == df["candidate_address"].str.strip().str.lower()
).astype(int)


# Country exact match
df["country_exact_match"] = (
    df["source1_country"].str.strip().str.lower()
    == df["candidate_country"].str.strip().str.lower()
).astype(int)


# Display matching features
print("\nMatching Features:")
print(
    df[
        [
            "source1_name",
            "candidate_name",
            "name_normalized_exact_match",
            "name_similarity",
            "address_exact_match",
            "country_exact_match"
        ]
    ].to_string(index=False)
)
print("\nTraining Data:")

print(
    df[
        [
            "source1_name",
            "candidate_name",
            "name_similarity",
            "address_exact_match",
            "country_exact_match",
            "match"
        ]
    ].to_string(index=False)
)