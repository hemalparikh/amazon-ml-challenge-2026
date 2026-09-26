import os
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from features import create_features


# -----------------------------
# Load training data
# -----------------------------

file_path = "data/sample_candidates.csv"
df = pd.read_csv(file_path)


# -----------------------------
# Create matching features
# -----------------------------

df = create_features(df)


# -----------------------------
# Select model features
# -----------------------------

features = [
    "name_normalized_exact_match",
    "name_similarity",
    "name_token_similarity",
    "name_first_token_match",
    "address_exact_match",
    "country_exact_match"
]

X = df[features]
y = df["match"]


print("\nFeatures used for training:")
print(X)

print("\nLabels:")
print(y)


# -----------------------------
# Train model
# -----------------------------

model = LogisticRegression()

model.fit(X, y)


# -----------------------------
# Evaluate on training data
# -----------------------------

predictions = model.predict(X)

accuracy = accuracy_score(y, predictions)

print("\nPredictions:")
print(predictions)

print("\nTraining accuracy:", accuracy)


# -----------------------------
# Match probabilities
# -----------------------------

probabilities = model.predict_proba(X)[:, 1]

df["match_probability"] = probabilities


print("\nMatch probabilities:")

print(
    df[
        [
            "source1_name",
            "candidate_name",
            "match",
            "match_probability"
        ]
    ].to_string(index=False)
)


# -----------------------------
# Save trained model
# -----------------------------

model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "matching_model.pkl")

joblib.dump(
    {
        "model": model,
        "features": features
    },
    model_path
)

print(f"\nModel saved successfully to: {model_path}")