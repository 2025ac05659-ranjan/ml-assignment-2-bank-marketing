from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
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


ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "model" / "model_bundle.pkl"


@st.cache_resource
def load_bundle():
    return joblib.load(BUNDLE_PATH)


def probability_or_score(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def metric_frame(y_true, y_pred, y_score):
    y_binary = (y_true == "yes").astype(int)
    try:
        auc = roc_auc_score(y_binary, y_score) if y_score is not None else None
    except Exception:
        auc = None

    return pd.DataFrame(
        [
            {
                "Accuracy": accuracy_score(y_true, y_pred),
                "AUC": auc,
                "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
                "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
                "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
                "MCC": matthews_corrcoef(y_true, y_pred),
            }
        ]
    ).round(4)


st.set_page_config(page_title="Bank Marketing Classifier", layout="wide")
st.title("Bank Marketing Classification")

bundle = load_bundle()
feature_columns = bundle["feature_columns"]
target = bundle["target"]
models = bundle["models"]

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload test CSV", type=["csv"])
    model_name = st.selectbox("Select model", list(models.keys()))

st.caption(
    "Dataset: UCI Bank Marketing. Target: predict whether a client subscribes to a term deposit."
)

if uploaded_file is None:
    st.info("Upload the provided test_data.csv file to view predictions and evaluation.")
    st.stop()

data = pd.read_csv(uploaded_file)
missing_columns = [col for col in feature_columns if col not in data.columns]

if missing_columns:
    st.error("The uploaded CSV is missing required feature columns.")
    st.write(missing_columns)
    st.stop()

X = data[feature_columns]
model = models[model_name]
y_pred = model.predict(X)
y_score = probability_or_score(model, X)

predictions = data.copy()
predictions["predicted_y"] = y_pred

st.subheader(model_name)

if target in data.columns:
    y_true = data[target]
    metrics = metric_frame(y_true, y_pred, y_score)
    st.dataframe(metrics, width="stretch", hide_index=True)

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_true, y_pred, labels=["no", "yes"])
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted no", "Predicted yes"],
            yticklabels=["Actual no", "Actual yes"],
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        st.pyplot(fig)

    with right:
        st.markdown("**Classification Report**")
        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(4), width="stretch")
else:
    st.warning("No target column found. Showing predictions only.")

st.markdown("**Prediction Preview**")
st.dataframe(predictions.head(50), width="stretch")

csv = predictions.to_csv(index=False).encode("utf-8")
st.download_button("Download predictions", csv, "predictions.csv", "text/csv")
