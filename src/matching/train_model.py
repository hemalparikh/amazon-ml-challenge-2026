import os
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    fbeta_score,
)

from features import create_features


# ============================================================
# CONFIGURATION
# ============================================================

TRAINING_FILE = "data/sample_candidates.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "matching_model.pkl")

DEFAULT_THRESHOLD = 0.50


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING TRAINING DATA")
print("=" * 70)

df = pd.read_csv(TRAINING_FILE)

print(f"Training rows: {len(df)}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# CREATE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("CREATING FEATURES")
print("=" * 70)

df = create_features(df)

print(f"Total columns after feature engineering: {len(df.columns)}")


# ============================================================
# MODEL FEATURES
# ============================================================

features = [
    # -------------------------
    # NAME FEATURES
    # -------------------------
    "name_normalized_exact_match",
    "name_similarity",
    "name_token_set_similarity",
    "name_token_sort_similarity",
    "name_first_token_match",
    "name_important_token_overlap",
    "name_important_token_similarity",
    "source1_name_token_count",
    "candidate_name_token_count",

    # -------------------------
    # ADDRESS FEATURES
    # -------------------------
    "address_exact_match",
    "address_similarity",
    "address_token_set_similarity",
    "address_token_sort_similarity",
    "address_token_overlap",
    "address_number_match",
    "address_postal_code_match",
    "source1_address_token_count",
    "candidate_address_token_count",

    # -------------------------
    # COUNTRY
    # -------------------------
    "country_exact_match",

    # -------------------------
    # MISSING VALUE FEATURES
    # -------------------------
    "source1_name_missing",
    "candidate_name_missing",
    "source1_address_missing",
    "candidate_address_missing",
    "source1_country_missing",
    "candidate_country_missing",
]


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in features
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"\nMissing features in dataframe:\n{missing_features}"
    )


X = df[features]
y = df["match"]


print("\nFeatures used for training:")
print(features)

print("\nFeature matrix:")
print(X.to_string(index=False))

print("\nLabels:")
print(y.to_string(index=False))


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION MODEL")
print("=" * 70)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)

model.fit(X, y)


# ============================================================
# PREDICTIONS
# ============================================================

probabilities = model.predict_proba(X)[:, 1]

predictions = (
    probabilities >= DEFAULT_THRESHOLD
).astype(int)


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD EVALUATION")
print("=" * 70)

thresholds = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]

for threshold in thresholds:

    threshold_predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y,
        threshold_predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        threshold_predictions,
        zero_division=0,
    )

    f05 = fbeta_score(
        y,
        threshold_predictions,
        beta=0.5,
        zero_division=0,
    )

    predicted_matches = threshold_predictions.sum()

    print(
        f"Threshold={threshold:.2f} | "
        f"Precision={precision:.3f} | "
        f"Recall={recall:.3f} | "
        f"F0.5={f05:.3f} | "
        f"Predicted Matches={predicted_matches}"
    )


# ============================================================
# DEFAULT THRESHOLD METRICS
# ============================================================

accuracy = accuracy_score(
    y,
    predictions,
)

precision = precision_score(
    y,
    predictions,
    zero_division=0,
)

recall = recall_score(
    y,
    predictions,
    zero_division=0,
)

f05 = fbeta_score(
    y,
    predictions,
    beta=0.5,
    zero_division=0,
)


print("\n" + "=" * 70)
print("DEFAULT THRESHOLD RESULTS")
print("=" * 70)

print(f"Threshold: {DEFAULT_THRESHOLD}")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F0.5:      {f05:.3f}")


# ============================================================
# DISPLAY PREDICTIONS
# ============================================================

df["match_probability"] = probabilities
df["predicted_match"] = predictions


print("\nPredictions:")

print(
    df[
        [
            "source1_name",
            "candidate_name",
            "match",
            "match_probability",
            "predicted_match",
        ]
    ].to_string(index=False)
)


# ============================================================
# MODEL COEFFICIENTS
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

importance = pd.DataFrame({
    "feature": features,
    "coefficient": model.coef_[0],
})

importance["absolute_coefficient"] = (
    importance["coefficient"].abs()
)

importance = importance.sort_values(
    "absolute_coefficient",
    ascending=False,
)

print(
    importance[
        ["feature", "coefficient"]
    ].to_string(index=False)
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True,
)

joblib.dump(
    {
        "model": model,
        "features": features,
        "threshold": DEFAULT_THRESHOLD,
    },
    MODEL_PATH,
)

print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(f"Model saved successfully to: {MODEL_PATH}")