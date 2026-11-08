"""
Credit Risk Classifier - Streamlit app for ML Assignment 2

Loads the five pipelines saved by model_training.ipynb (model/*.joblib) and evaluates them on an
uploaded test CSV. Nothing is retrained here - the app only calls predict() / predict_proba().
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_CSV = BASE_DIR / "test_data.csv"
TARGET = "credit_risk"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

MODEL_NOTES = {
    "Logistic Regression": "Linear model on one-hot + scaled features, C=10, class_weight='balanced'.",
    "Decision Tree": "Gini tree, max_depth=4, min_samples_leaf=20, class_weight='balanced'.",
    "KNN": "k=3 neighbours, uniform weights, Euclidean distance on scaled features.",
    "Naive Bayes": "Gaussian NB, var_smoothing=1e-3.",
    "Support Vector Machine": "RBF kernel, C=1, gamma=0.01, class_weight='balanced', probability=True.",
}

FEATURES = ["checking_account", "duration_months", "credit_history", "purpose", "credit_amount",
            "savings_account", "employment_since", "installment_rate", "personal_status_sex",
            "other_debtors", "residence_since", "property", "age", "other_installment_plans",
            "housing", "existing_credits", "job", "num_dependents", "telephone", "foreign_worker"]

METRIC_HELP = {
    "Accuracy": "Share of all applicants classified correctly.",
    "AUC": "Area under the ROC curve, from the predicted probability of 'bad'. 0.5 = random guessing.",
    "Precision": "Of the applicants flagged as bad, how many really were bad.",
    "Recall": "Of the applicants who really were bad, how many the model caught.",
    "F1": "Harmonic mean of precision and recall (bad class).",
    "MCC": "Matthews correlation, uses all four cells of the confusion matrix. 0 = no better than chance.",
}


st.set_page_config(page_title="Credit Risk Classifier", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 2rem;}
    div[data-testid="stMetric"] {background: rgba(46, 94, 170, 0.07); border-radius: 8px;
                                 padding: 10px 14px; border-left: 4px solid #2E5EAA;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}


def to_binary_target(series):
    """Accepts good/bad labels (as in test_data.csv) or 0/1 (1 = bad). Returns 0/1 or None if unknown."""
    if pd.api.types.is_numeric_dtype(series):
        values = set(series.dropna().unique())
        return series.astype(int) if values <= {0, 1} else None
    s = series.astype(str).str.strip().str.lower()
    if set(s.unique()) <= {"good", "bad"}:
        return (s == "bad").astype(int)
    return None


def score(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob) if y.nunique() == 2 else float("nan"),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return metrics, y_pred, y_prob


# ---------------- sidebar ----------------
st.sidebar.header("1. Test data")
uploaded = st.sidebar.file_uploader("Upload test CSV", type=["csv"],
                                    help="Must have the 20 input columns and the credit_risk column.")
use_sample = st.sidebar.checkbox("No file? Use test_data.csv from the repo", value=False)
if SAMPLE_CSV.exists():
    st.sidebar.download_button("Download test_data.csv", SAMPLE_CSV.read_bytes(),
                               file_name="test_data.csv", mime="text/csv")

st.sidebar.header("2. Model")
model_name = st.sidebar.selectbox("Choose a classifier", list(MODEL_FILES.keys()), index=4)
st.sidebar.caption(MODEL_NOTES[model_name])
st.sidebar.divider()
st.sidebar.caption("Models were trained once in model_training.ipynb and loaded here from model/. "
                   "Uploaded data is only used for prediction, never for training.")

# ---------------- header ----------------
st.title("Credit Risk Classifier")
st.write("Predicts whether a loan applicant is a **good** or **bad** credit risk "
         "(UCI Statlog German Credit data, 20 features). "
         "*Bad* is treated as the positive class, so precision, recall and F1 are for the bad class.")

with st.expander("Expected CSV format"):
    st.write("One row per applicant. Required columns (any order, extra columns are ignored):")
    st.code(", ".join(FEATURES + [TARGET]), language=None)
    st.write("`credit_risk` can be `good`/`bad` or `0`/`1` (1 = bad).")

# ---------------- load data ----------------
if uploaded is not None:
    data = pd.read_csv(uploaded)
    source = uploaded.name
elif use_sample and SAMPLE_CSV.exists():
    data = pd.read_csv(SAMPLE_CSV)
    source = "test_data.csv (from repo)"
else:
    st.info("Upload a test CSV in the sidebar (or tick the sample-data box) to see the results.")
    st.stop()

missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
if missing:
    st.error(f"The file is missing {len(missing)} required column(s): {', '.join(missing)}")
    st.stop()

data = data.dropna(subset=FEATURES + [TARGET])
y_true = to_binary_target(data[TARGET])
if y_true is None:
    st.error("`credit_risk` must contain only good/bad or 0/1 values.")
    st.stop()
X = data[FEATURES]

models = load_models()
st.success(f"Loaded **{source}**: {len(data)} rows, {int(y_true.sum())} bad / {int((1 - y_true).sum())} good.")

tab_model, tab_compare, tab_pred = st.tabs([f"{model_name}", "Compare all 5 models", "Predictions"])

# ---------------- selected model ----------------
with tab_model:
    metrics, y_pred, y_prob = score(models[model_name], X, y_true)

    cols = st.columns(6)
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, "n/a" if np.isnan(value) else f"{value:.4f}", help=METRIC_HELP[label])

    left, right = st.columns([1, 1.2])
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["good", "bad"], yticklabels=["good", "bad"], annot_kws={"size": 14})
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"{tp} bad applicants caught, {fn} missed (approved by mistake), "
                   f"{fp} good applicants wrongly flagged.")
    with right:
        st.subheader("Classification report")
        report = classification_report(y_true, y_pred, labels=[0, 1], target_names=["good", "bad"],
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

# ---------------- comparison ----------------
with tab_compare:
    rows = {name: score(m, X, y_true)[0] for name, m in models.items()}
    table = pd.DataFrame(rows).T
    table.index.name = "ML Model Name"
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#cfe3ff"),
                 width="stretch")
    st.caption("Highlighted = best value in each column on the uploaded data.")

    fig, ax = plt.subplots(figsize=(10, 3.4))
    table[["Accuracy", "AUC", "F1", "MCC"]].plot(kind="bar", ax=ax, rot=0, width=0.75)
    ax.set_ylim(0, 1)
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n in table.index], fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False, fontsize=9)
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

# ---------------- predictions ----------------
with tab_pred:
    out = data.copy()
    out["predicted"] = np.where(y_pred == 1, "bad", "good")
    out["prob_bad"] = np.round(y_prob, 4)
    out["correct"] = y_pred == y_true.values
    st.write(f"Row-level predictions from **{model_name}**:")
    st.dataframe(out, width="stretch", height=380)
    st.download_button("Download predictions as CSV", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{MODEL_FILES[model_name].replace('.joblib', '')}.csv",
                       mime="text/csv")
