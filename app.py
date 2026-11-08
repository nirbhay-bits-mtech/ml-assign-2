"""
Student Outcome Predictor - Streamlit app (ML Assignment 2)

Loads the five pre-trained classifiers from model/ and evaluates them on a
user-uploaded test CSV. No retraining happens inside the app.
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
SAMPLE_TEST = os.path.join(BASE_DIR, "test_data.csv")

st.set_page_config(page_title="Student Outcome Predictor", page_icon="🎓", layout="wide")


# --------------------------------------------------------------------------
# loading helpers
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open(os.path.join(MODEL_DIR, "metadata.json")) as fh:
        meta = json.load(fh)
    encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    models = {
        name: joblib.load(os.path.join(MODEL_DIR, fname))
        for name, fname in meta["model_files"].items()
    }
    return meta, encoder, models


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def encode_target(series: pd.Series, encoder) -> np.ndarray:
    """Accept either text labels (Dropout/Enrolled/Graduate) or 0/1/2 codes."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int).to_numpy()
    return encoder.transform(series.astype(str).str.strip())


def compute_metrics(y_true, y_pred, y_prob, n_classes) -> dict:
    try:
        auc = roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="weighted", labels=list(range(n_classes))
        )
    except ValueError:  # e.g. a class missing from the uploaded file
        auc = np.nan
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def plot_confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
try:
    meta, encoder, models = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found in `model/`. Run `python train_models.py` first.")
    st.stop()

classes = meta["classes"]
feature_cols = meta["feature_columns"]
target_col = meta["target"]

st.title("🎓 Student Outcome Predictor")
st.caption(
    "Predicts whether a student will **Drop out**, stay **Enrolled** or **Graduate** "
    "using 5 classifiers trained on the UCI *Predict Students' Dropout and Academic Success* dataset."
)

with st.sidebar:
    st.header("⚙️ Controls")
    model_name = st.selectbox("Choose a model", list(models.keys()))
    uploaded = st.file_uploader("Upload test CSV", type=["csv"])
    use_sample = st.checkbox("Use bundled test_data.csv instead", value=uploaded is None)

    if os.path.exists(SAMPLE_TEST):
        with open(SAMPLE_TEST, "rb") as fh:
            st.download_button("⬇️ Download sample test_data.csv", fh, file_name="test_data.csv")

    st.divider()
    st.markdown(f"**Features expected:** {len(feature_cols)}  \n**Target column:** `{target_col}`")
    st.markdown("Models are pre-trained; the app only runs inference.")

# --- pick data source
if uploaded is not None and not use_sample:
    data = clean_columns(pd.read_csv(uploaded))
    source = uploaded.name
elif use_sample and os.path.exists(SAMPLE_TEST):
    data = clean_columns(pd.read_csv(SAMPLE_TEST))
    source = "test_data.csv (bundled)"
else:
    st.info("👈 Upload a test CSV (or tick *Use bundled test_data.csv*) to begin.")
    st.stop()

missing = [c for c in feature_cols if c not in data.columns]
if missing:
    st.error(f"Uploaded file is missing {len(missing)} required column(s):")
    st.write(missing)
    st.stop()

X = data[feature_cols].astype(float)
has_target = target_col in data.columns

st.subheader(f"📄 Data: {source}")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", len(data))
c2.metric("Features used", len(feature_cols))
c3.metric("Target present", "Yes" if has_target else "No")
with st.expander("Preview uploaded data"):
    st.dataframe(data.head(20), use_container_width=True)

model = models[model_name]
y_pred = model.predict(X)
y_prob = model.predict_proba(X)

tab_eval, tab_compare, tab_pred, tab_info = st.tabs(
    ["📊 Evaluation", "🏆 Compare all models", "🔮 Predictions", "ℹ️ Model info"]
)

# --- evaluation of selected model
with tab_eval:
    if not has_target:
        st.warning(f"No `{target_col}` column found - metrics need the true labels.")
    else:
        try:
            y_true = encode_target(data[target_col], encoder)
        except ValueError as exc:
            st.error(f"Could not read target labels: {exc}")
            st.stop()

        m = compute_metrics(y_true, y_pred, y_prob, len(classes))
        st.markdown(f"### {model_name}")
        cols = st.columns(6)
        for col, (k, v) in zip(cols, m.items()):
            col.metric(k, "N/A" if pd.isna(v) else f"{v:.4f}")
        st.caption("Precision / Recall / F1 are weighted averages; AUC is one-vs-rest weighted.")

        left, right = st.columns([1, 1.2])
        with left:
            st.markdown("**Confusion matrix**")
            st.pyplot(plot_confusion(y_true, y_pred, classes))
        with right:
            st.markdown("**Classification report**")
            report = classification_report(
                y_true, y_pred, labels=list(range(len(classes))),
                target_names=classes, output_dict=True, zero_division=0,
            )
            st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

# --- compare all five models on the same uploaded data
with tab_compare:
    if not has_target:
        st.warning("Comparison needs the target column.")
    else:
        y_true = encode_target(data[target_col], encoder)
        table = {}
        for name, mdl in models.items():
            table[name] = compute_metrics(y_true, mdl.predict(X), mdl.predict_proba(X), len(classes))
        comp = pd.DataFrame(table).T
        st.dataframe(
            comp.style.format("{:.4f}").highlight_max(axis=0, color="#c8e6c9"),
            use_container_width=True,
        )
        metric_pick = st.radio("Plot metric", list(comp.columns), horizontal=True, index=4)
        fig, ax = plt.subplots(figsize=(8, 3.2))
        comp[metric_pick].sort_values().plot.barh(ax=ax, color="#4c72b0")
        ax.set_xlabel(metric_pick)
        ax.set_xlim(0, 1)
        fig.tight_layout()
        st.pyplot(fig)
        st.success(f"Best on this file by F1: **{comp['F1'].idxmax()}**")

# --- row-level predictions
with tab_pred:
    out = data.copy()
    out["Predicted"] = encoder.inverse_transform(y_pred)
    out["Confidence"] = y_prob.max(axis=1).round(3)
    st.dataframe(out[[*([target_col] if has_target else []), "Predicted", "Confidence"]].head(200),
                 use_container_width=True)
    st.download_button("⬇️ Download predictions", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{model_name.replace(' ', '_').lower()}.csv")

# --- model info
with tab_info:
    st.json({"model": model_name, "best_params": meta["best_params"].get(model_name, {})})
    st.markdown(
        f"Trained on {meta['train_rows']} rows, held-out test {meta['test_rows']} rows "
        f"(stratified 80/20). scikit-learn {meta['sklearn']}."
    )
    st.code(str(model), language="text")
