"""
ML Assignment 2 - Training script
Dataset : Predict Students' Dropout and Academic Success (UCI id 697)
Task    : 3-class classification  ->  Dropout / Enrolled / Graduate

What this script does
1. Loads the dataset (local copy in data/ or downloads it from UCI)
2. Stratified 80/20 train-test split
3. Trains 5 classifiers (LR, DT, KNN, Gaussian NB, SVM) with small grid searches
4. Evaluates each on the held-out test set:
   Accuracy, AUC (one-vs-rest, weighted), Precision, Recall, F1 (weighted), MCC
5. Saves fitted pipelines to model/*.joblib, the test split to test_data.csv,
   metrics to model/metrics.csv and writes the comparison table into README.md

Run:  python train_models.py
"""

import io
import json
import os
import platform
import urllib.request
import zipfile

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
LOCAL_CSV = os.path.join(DATA_DIR, "students_dropout.csv")
UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/697/"
    "predict+students+dropout+and+academic+success.zip"
)
TARGET = "Target"

# file names used by app.py as well
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


# --------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------
def load_dataset() -> pd.DataFrame:
    """Read local CSV if present, else download the UCI zip and cache it."""
    if os.path.exists(LOCAL_CSV):
        print(f"[data] using local file {LOCAL_CSV}")
        df = pd.read_csv(LOCAL_CSV)
    else:
        print(f"[data] downloading from {UCI_ZIP_URL}")
        with urllib.request.urlopen(UCI_ZIP_URL, timeout=60) as resp:
            raw = resp.read()
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
            df = pd.read_csv(zf.open(csv_name), sep=";")
        os.makedirs(DATA_DIR, exist_ok=True)
        df.columns = [c.strip() for c in df.columns]
        df.to_csv(LOCAL_CSV, index=False)
        print(f"[data] cached to {LOCAL_CSV}")

    # original header has stray tabs/spaces (e.g. 'Daytime/evening attendance\t')
    df.columns = [c.strip() for c in df.columns]
    return df


def describe(df: pd.DataFrame) -> None:
    print(f"[data] shape = {df.shape}  (features = {df.shape[1] - 1})")
    print(f"[data] missing values = {int(df.isna().sum().sum())}")
    print("[data] class distribution:")
    print(df[TARGET].value_counts().to_string())


# --------------------------------------------------------------------------
# 2. Models  (each is a Pipeline so scaling is stored with the model)
# --------------------------------------------------------------------------
def build_search_spaces():
    return {
        "Logistic Regression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, random_state=SEED)),
            ]),
            {"clf__C": [0.05, 0.1, 0.5, 1.0, 5.0]},
        ),
        "Decision Tree": (
            Pipeline([
                ("clf", DecisionTreeClassifier(random_state=SEED)),
            ]),
            {
                "clf__max_depth": [4, 6, 8, 10, None],
                "clf__min_samples_leaf": [1, 5, 10, 20],
                "clf__criterion": ["gini", "entropy"],
            },
        ),
        "KNN": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier()),
            ]),
            {
                "clf__n_neighbors": [5, 9, 15, 21, 31],
                "clf__weights": ["uniform", "distance"],
            },
        ),
        "Naive Bayes": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GaussianNB()),
            ]),
            {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]},
        ),
        "Support Vector Machine": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, random_state=SEED)),
            ]),
            {"clf__C": [0.5, 1.0, 5.0], "clf__gamma": ["scale", 0.01]},
        ),
    }


# --------------------------------------------------------------------------
# 3. Metrics
# --------------------------------------------------------------------------
def evaluate(model, X, y_true) -> dict:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def metrics_to_markdown(results: pd.DataFrame) -> str:
    lines = [
        "| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, row in results.iterrows():
        lines.append(
            f"| {name} | {row['Accuracy']:.4f} | {row['AUC']:.4f} | {row['Precision']:.4f} "
            f"| {row['Recall']:.4f} | {row['F1']:.4f} | {row['MCC']:.4f} |"
        )
    return "\n".join(lines)


def write_table_into_readme(table_md: str) -> None:
    readme = os.path.join(BASE_DIR, "README.md")
    start, end = "<!-- METRICS_TABLE_START -->", "<!-- METRICS_TABLE_END -->"
    if not os.path.exists(readme):
        return
    with open(readme, encoding="utf-8") as fh:
        text = fh.read()
    if start in text and end in text:
        head, rest = text.split(start, 1)
        _, tail = rest.split(end, 1)
        text = f"{head}{start}\n{table_md}\n{end}{tail}"
        with open(readme, "w", encoding="utf-8") as fh:
            fh.write(text)
        print("[readme] comparison table updated")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    df = load_dataset()
    describe(df)

    feature_cols = [c for c in df.columns if c != TARGET]
    X = df[feature_cols].astype(float)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[TARGET])
    print(f"[data] label mapping = {dict(zip(encoder.classes_, range(len(encoder.classes_))))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    print(f"[split] train = {X_train.shape}, test = {X_test.shape}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rows, best_params = {}, {}
    for name, (pipe, grid) in build_search_spaces().items():
        print(f"\n[train] {name} ...")
        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params[name] = {k.replace("clf__", ""): str(v) for k, v in search.best_params_.items()}
        print(f"        best params = {best_params[name]}  (cv macro-F1 = {search.best_score_:.4f})")

        rows[name] = evaluate(model, X_test, y_test)
        print("        test metrics = " + ", ".join(f"{k}={v:.4f}" for k, v in rows[name].items()))
        print(classification_report(y_test, model.predict(X_test), target_names=encoder.classes_))

        joblib.dump(model, os.path.join(MODEL_DIR, MODEL_FILES[name]))

    results = pd.DataFrame(rows).T[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    results.index.name = "Model"
    results.round(4).to_csv(os.path.join(MODEL_DIR, "metrics.csv"))

    # artefacts needed by the Streamlit app
    joblib.dump(encoder, os.path.join(MODEL_DIR, "label_encoder.joblib"))
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as fh:
        json.dump(
            {
                "feature_columns": feature_cols,
                "target": TARGET,
                "classes": encoder.classes_.tolist(),
                "model_files": MODEL_FILES,
                "best_params": best_params,
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "python": platform.python_version(),
                "sklearn": sklearn.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            fh,
            indent=2,
        )

    # held-out test split (features + string target) for uploading in the app
    test_df = X_test.copy()
    test_df[TARGET] = encoder.inverse_transform(y_test)
    test_df.to_csv(os.path.join(BASE_DIR, "test_data.csv"), index=False)
    print(f"\n[save] test_data.csv written ({len(test_df)} rows)")

    table_md = metrics_to_markdown(results)
    print("\n" + table_md)
    write_table_into_readme(table_md)
    print(f"\nDone. sklearn {sklearn.__version__}, python {platform.python_version()}")


if __name__ == "__main__":
    main()
