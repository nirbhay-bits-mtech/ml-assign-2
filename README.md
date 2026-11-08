# 🎓 Student Outcome Predictor — ML Assignment 2

**M.Tech AIML (BITS Pilani WILP) · Machine Learning · Assignment 2**
Author: Nirbhay Jha

🔗 **Live app:** _<STREAMLIT_APP_URL>_
🔗 **GitHub repo:** https://github.com/nirbhay-bits-mtech/ml-assign-2

---

## a. Problem statement

Higher-education institutions lose a significant share of students to dropout, and early identification of at-risk students lets them intervene (mentoring, scholarships, academic support). The goal is to **predict the final academic status of a student — `Dropout`, `Enrolled` or `Graduate` —** from information available at enrolment plus first- and second-semester performance.

This is a **3-class, imbalanced classification problem**. Five classical classifiers are trained, compared on six evaluation metrics, and served through an interactive Streamlit app that evaluates the saved models on an uploaded test CSV (no retraining in the app).

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Predict Students' Dropout and Academic Success |
| Source | UCI Machine Learning Repository, id 697 — https://archive.ics.uci.edu/dataset/697 |
| Citation | Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). DOI: 10.24432/C5MC89 (CC BY 4.0) |
| Instances | 4,424 students |
| Features | 36 (integer-coded categorical + continuous) |
| Target | `Target` ∈ {Dropout, Enrolled, Graduate} |
| Class balance | Graduate ≈ 50 %, Dropout ≈ 32 %, Enrolled ≈ 18 % |
| Missing values | None |

**Feature groups**

- **Demographic:** marital status, nationality, gender, age at enrolment, displaced, international
- **Socio-economic:** mother's/father's qualification and occupation, scholarship holder, debtor, tuition fees up to date, educational special needs
- **Academic path at enrolment:** application mode/order, course, daytime/evening attendance, previous qualification (+ grade), admission grade
- **Academic performance:** curricular units credited / enrolled / evaluated / approved / grade / without evaluation — for 1st and 2nd semester (12 features)
- **Macro-economic:** unemployment rate, inflation rate, GDP

**Pre-processing**

- Column names stripped of stray whitespace (the raw header contains tab characters).
- Target label-encoded (Dropout = 0, Enrolled = 1, Graduate = 2); encoder saved to `model/label_encoder.joblib`.
- Stratified 80/20 train–test split (`random_state = 42`) → 3,539 train / 885 test rows. The test split is saved as `test_data.csv`.
- `StandardScaler` inside each pipeline for scale-sensitive models (LR, KNN, SVM, NB). The scaler is fitted only on the training data and saved with the model.
- Hyper-parameters chosen with 5-fold stratified `GridSearchCV` on the training set, optimising macro-F1 (so the minority `Enrolled` class is not ignored).

## c. GitHub repository

https://github.com/nirbhay-bits-mtech/ml-assign-2

```
ml-assign-2/
├── app.py                 # Streamlit app (inference + evaluation only)
├── train_models.py        # data loading, training, evaluation, saving
├── requirements.txt
├── README.md
├── test_data.csv          # held-out 20% test split (features + Target)
├── data/
│   └── students_dropout.csv
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   ├── label_encoder.joblib
│   ├── metadata.json      # feature list, classes, best params, versions
│   └── metrics.csv        # test-set metrics for all models
└── .streamlit/config.toml # app theme
```

**Reproduce**

```bash
pip install -r requirements.txt
python train_models.py      # trains, saves models, writes test_data.csv and the table below
streamlit run app.py
```

## d. Models used and evaluation

Models: Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive Bayes, Support Vector Machine (RBF kernel).

Metrics are computed on the held-out test set (885 rows). Precision, Recall and F1 are **weighted averages** across the three classes; AUC is **one-vs-rest, weighted**; MCC is the multi-class Matthews correlation coefficient.

### Comparison table

<!-- METRICS_TABLE_START -->
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | _run train_models.py_ | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |
<!-- METRICS_TABLE_END -->

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | _to be finalised after training_ |
| Decision Tree | |
| KNN | |
| Naive Bayes | |
| Support Vector Machine | |
| **Overall winner for this dataset** | |

## Streamlit app features

- **CSV upload** of test data (or one-click use of the bundled `test_data.csv`, also downloadable)
- **Model selection dropdown** for the five saved models
- **Evaluation metrics** — Accuracy, AUC, Precision, Recall, F1, MCC
- **Confusion matrix** heat-map and **classification report**
- **Compare all models** tab — side-by-side metric table with best values highlighted + bar chart
- **Predictions** tab with per-row confidence and CSV download
- Column validation with a clear error if required features are missing
