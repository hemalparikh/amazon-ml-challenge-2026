import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, fbeta_score


TRAIN_PATH = "data/matching/train_features.csv"
VAL_PATH = "data/matching/val_features.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


DROP_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
    "source1_name",
    "candidate_name",
    "source1_address",
    "candidate_address",
    "source1_country",
    "candidate_country",
    "source1_name_normalized",
    "candidate_name_normalized",
    "source1_address_normalized",
    "candidate_address_normalized",
    "source1_country_normalized",
    "candidate_country_normalized",
    "label",
]


def main():

    print("=" * 60)
    print("TRAINING REAL ENTITY MATCHING MODEL")
    print("=" * 60)

    print("\nLoading training data...")
    train = pd.read_csv(TRAIN_PATH)

    print("Loading validation data...")
    val = pd.read_csv(VAL_PATH)

    feature_columns = [
        c for c in train.columns
        if c not in DROP_COLUMNS
    ]

    print(f"\nFeatures: {len(feature_columns)}")
    print(feature_columns)

    X_train = train[feature_columns]
    y_train = train["label"]

    X_val = val[feature_columns]
    y_val = val["label"]

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("Training complete.")

    probabilities = model.predict_proba(X_val)[:, 1]

    print("\n" + "=" * 60)
    print("THRESHOLD EVALUATION")
    print("=" * 60)

    best_threshold = None
    best_f05 = -1

    for threshold in [
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
    ]:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_val,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_val,
            predictions,
            zero_division=0,
        )

        f05 = fbeta_score(
            y_val,
            predictions,
            beta=0.5,
            zero_division=0,
        )

        print(
            f"threshold={threshold:.2f} "
            f"precision={precision:.4f} "
            f"recall={recall:.4f} "
            f"F0.5={f05:.4f}"
        )

        if f05 > best_f05:
            best_f05 = f05
            best_threshold = threshold

    print("\nBest threshold:", best_threshold)
    print("Best validation F0.5:", round(best_f05, 4))

    print("\nFeature importance:")

    importance = pd.Series(
        model.feature_importances_,
        index=feature_columns,
    ).sort_values(ascending=False)

    print(importance.head(15).to_string())

    model_path = os.path.join(
        MODEL_DIR,
        "matching_model_real.pkl",
    )

    joblib.dump(
        {
            "model": model,
            "threshold": best_threshold,
            "feature_columns": feature_columns,
        },
        model_path,
    )

    print("\nModel saved:")
    print(model_path)


if __name__ == "__main__":
    main()