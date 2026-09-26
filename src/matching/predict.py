import pandas as pd
import joblib

from features import create_features


# -----------------------------
# Load candidate pairs
# -----------------------------

file_path = "data/sample_new_candidates.csv"
df = pd.read_csv(file_path)


# -----------------------------
# Create matching features
# -----------------------------

df = create_features(df)


# -----------------------------
# Load trained model
# -----------------------------

model_data = joblib.load("models/matching_model.pkl")

model = model_data["model"]
features = model_data["features"]


# -----------------------------
# Make predictions
# -----------------------------

X = df[features]

df["match_probability"] = model.predict_proba(X)[:, 1]

df["predicted_match"] = model.predict(X)


# -----------------------------
# Display predictions
# -----------------------------

print("\nMatching Predictions:")

print(
    df[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "source1_name",
            "candidate_name",
            "match_probability",
            "predicted_match"
        ]
    ].to_string(index=False)
)