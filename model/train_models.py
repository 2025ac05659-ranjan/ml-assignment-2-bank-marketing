from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "bank-additional-full.csv"
MODEL_DIR = ROOT / "model"
OUTPUT_DIR = ROOT / "outputs"
TEST_DATA_PATH = ROOT / "test_data.csv"
TARGET = "y"
POSITIVE_LABEL = "yes"
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=";")
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def probability_or_score(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def evaluate_model(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_binary = (y_test == POSITIVE_LABEL).astype(int)

    try:
        score_values = probability_or_score(model, X_test)
        auc = roc_auc_score(y_binary, score_values)
    except Exception:
        auc = np.nan

    return {
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=["no", "yes"]).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    test_export = X_test.copy()
    test_export[TARGET] = y_test.values
    test_export.sample(n=min(750, len(test_export)), random_state=RANDOM_STATE).to_csv(
        TEST_DATA_PATH, index=False
    )

    preprocessor = build_preprocessor(X)
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "kNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    metrics = []
    bundle = {
        "target": TARGET,
        "positive_label": POSITIVE_LABEL,
        "feature_columns": X.columns.tolist(),
        "models": {},
    }

    for name, estimator in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        result = evaluate_model(name, pipeline, X_test, y_test)
        metrics.append(result)
        bundle["models"][name] = pipeline

        safe_name = name.lower().replace(" ", "_").replace("/", "_")
        joblib.dump(pipeline, MODEL_DIR / f"{safe_name}.pkl")

    joblib.dump(bundle, MODEL_DIR / "model_bundle.pkl")

    metrics_table = pd.DataFrame(
        [
            {k: v for k, v in row.items() if k not in {"confusion_matrix", "classification_report"}}
            for row in metrics
        ]
    )
    metric_columns = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    metrics_table[metric_columns] = metrics_table[metric_columns].round(4)
    metrics_table.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    with open(OUTPUT_DIR / "evaluation_details.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(metrics_table.to_string(index=False))
    print(f"Saved test data: {TEST_DATA_PATH}")
    print(f"Saved models: {MODEL_DIR}")


if __name__ == "__main__":
    main()
