import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Page configuration
st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)
# Custom Sage Green Theme
st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #F1F5F0;
}

/* Header */
[data-testid="stHeader"] {
    background-color: #F1F5F0;
}

/* Main headings */
h1, h2, h3 {
    color: #354C3B !important;
}

/* Normal text */
.stApp p, .stApp label {
    color: #354C3B;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #DCE6DC;
    padding: 18px;
    border-radius: 12px;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #FFFFFF;
    border: 1px dashed #9BB49E;
    border-radius: 12px;
    padding: 12px;
}

/* Select box */
[data-testid="stSelectbox"] {
    background-color: #FFFFFF;
    border-radius: 8px;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    background-color: #FFFFFF;
    border-radius: 10px;
}

/* Buttons */
.stButton > button {
    background-color: #64876D;
    color: white;
    border-radius: 8px;
    border: none;
}

.stButton > button:hover {
    background-color: #4F7058;
    color: white;
}

/* Divider */
hr {
    border-color: #DCE6DC;
}

</style>
""", unsafe_allow_html=True)

# Application title
st.title("🌱 Dry Bean Classification")
st.write(
    "An interactive Machine Learning application "
    "for evaluating five classification models."
)

st.divider()

# Model directory
MODEL_DIR = "models"

# Model file mapping
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

@st.cache_resource
def load_models():

    loaded_models = {}

    for name, filename in MODEL_FILES.items():

        path = os.path.join(MODEL_DIR, filename)

        loaded_models[name] = joblib.load(path)

    encoder_path = os.path.join(
        MODEL_DIR,
        "label_encoder.joblib"
    )

    loaded_encoder = joblib.load(encoder_path)

    return loaded_models, loaded_encoder


try:
    models, label_encoder = load_models()

except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    test_df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    st.write("Dataset dimensions:", test_df.shape)

    st.subheader("Dataset Preview")

    st.dataframe(test_df.head(), use_container_width=True)

else:

    st.info("Please upload a test CSV.")
    st.stop()


# Validate required columns
expected_features = list(models["Logistic Regression"].feature_names_in_)

required_columns = expected_features + ["Class"]

missing_columns = [
    col for col in required_columns
    if col not in test_df.columns
]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

# Separate features and actual target
X_test = test_df[expected_features]
y_actual = test_df["Class"]

if X_test.isnull().any().any() or y_actual.isnull().any():
    st.error("The uploaded dataset contains missing values.")
    st.stop()

try:
    X_test = X_test.astype(float)
    y_actual_encoded = label_encoder.transform(y_actual)

except (ValueError, TypeError):
    st.error("Invalid feature values or unknown target classes.")
    st.stop()

st.success("Dataset validation completed!")


st.divider()

st.header("2. Select Classification Model")

selected_model_name = st.selectbox(
    "Choose a model",
    list(models.keys())
)

selected_model = models[selected_model_name]

st.write(f"Selected model: **{selected_model_name}**")

# Predict using the already-trained model
y_pred = selected_model.predict(X_test)

y_prob = selected_model.predict_proba(X_test)


st.divider()

st.header("3. Model Evaluation Metrics")

accuracy = accuracy_score(y_actual_encoded, y_pred)

auc = roc_auc_score(
    y_actual_encoded,
    y_prob,
    multi_class="ovr",
    average="macro",
    labels=np.arange(len(label_encoder.classes_))
)

precision = precision_score(
    y_actual_encoded,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_actual_encoded,
    y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_actual_encoded,
    y_pred,
    average="macro",
    zero_division=0
)

mcc = matthews_corrcoef(y_actual_encoded, y_pred)

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", f"{accuracy:.4f}")
col2.metric("AUC Score", f"{auc:.4f}")
col3.metric("Precision", f"{precision:.4f}")

col4, col5, col6 = st.columns(3)

col4.metric("Recall", f"{recall:.4f}")
col5.metric("F1 Score", f"{f1:.4f}")
col6.metric("MCC Score", f"{mcc:.4f}")

st.divider()

st.header("4. Confusion Matrix")

cm = confusion_matrix(
    y_actual_encoded,
    y_pred,
    labels=np.arange(len(label_encoder.classes_))
)

fig, ax = plt.subplots(figsize=(10, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
    ax=ax
)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")
ax.set_title(f"Confusion Matrix - {selected_model_name}")

plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.divider()

st.header("5. Classification Report")

report = classification_report(
    y_actual_encoded,
    y_pred,
    labels=np.arange(len(label_encoder.classes_)),
    target_names=label_encoder.classes_,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)
