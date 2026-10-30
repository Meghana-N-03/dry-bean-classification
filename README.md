# Dry Bean Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to develop and evaluate multiple Machine Learning classification models to classify dry bean varieties based on their physical and geometrical characteristics.

The project implements five supervised Machine Learning classification algorithms and compares their performance using different evaluation metrics.

The five classification algorithms implemented are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The models are evaluated using the following performance metrics:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

An interactive web application is also developed using Streamlit. It allows users to upload a test dataset, select a trained classification model, and view its evaluation metrics, confusion matrix, and classification report.

## 2. Dataset Description

**Dataset Name:** Dry Bean Dataset

**Source:** UCI Machine Learning Repository

**Dataset Link:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

The Dry Bean Dataset contains measurements of different dry bean varieties. The dataset consists of numerical features describing the shape, size, and geometrical properties of dry beans.

### Dataset Details

| Property | Description |
|---|---|
| Dataset Name | Dry Bean Dataset |
| Total Instances | 13,611 |
| Total Input Features | 16 |
| Target Variable | Class |
| Number of Classes | 7 |
| Problem Type | Multiclass Classification |
| Missing Values | 0 |

### Input Features

The dataset contains the following 16 input features:

1. Area
2. Perimeter
3. MajorAxisLength
4. MinorAxisLength
5. AspectRation
6. Eccentricity
7. ConvexArea
8. EquivDiameter
9. Extent
10. Solidity
11. roundness
12. Compactness
13. ShapeFactor1
14. ShapeFactor2
15. ShapeFactor3
16. ShapeFactor4

### Target Classes

The target variable `Class` contains seven dry bean varieties:

- BARBUNYA
- BOMBAY
- CALI
- DERMASON
- HOROZ
- SEKER
- SIRA

### Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Checked the dataset dimensions, data types, and missing values.
3. Separated the independent features and target variable.
4. Encoded the categorical target variable using LabelEncoder.
5. Split the dataset into training and testing sets using an 80:20 ratio.
6. Used stratified sampling to maintain class proportions in both datasets.

**Train-Test Split:**

- Training instances: 10,888
- Testing instances: 2,723
- Random State: 42

StandardScaler was applied to Logistic Regression, KNN, and SVM using Scikit-learn Pipelines.

Decision Tree and Gaussian Naive Bayes were implemented without feature scaling.

## 3. GitHub Repository Link

**GitHub Repository:** [INSERT YOUR GITHUB REPOSITORY LINK HERE]

The repository contains:

- `app.py` – Streamlit application source code.
- `requirements.txt` – Required Python dependencies.
- `README.md` – Project documentation.
- `test_data.csv` – Test dataset used for model evaluation.
- `model_comparison.csv` – Comparison of evaluation results.
- `models/` – Saved trained models and label encoder.
- `notebooks/` – Jupyter Notebook containing the complete implementation and experiments.

## 4. Models Used and Evaluation Metrics

All five classification models were trained and evaluated using the same training and testing datasets.

The following evaluation metrics were calculated:

- **Accuracy:** Measures the proportion of correctly classified instances.
- **AUC Score:** Measures the ability of the model to distinguish between different classes using a One-vs-Rest approach.
- **Precision:** Measures the correctness of positive predictions.
- **Recall:** Measures the proportion of actual instances correctly identified.
- **F1 Score:** Represents the harmonic mean of precision and recall.
- **MCC:** Measures the overall quality of classification predictions using all confusion matrix outcomes.

Macro averaging was used for AUC, Precision, Recall, and F1 Score to give equal importance to all seven classes.

### Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9214 | 0.9948 | 0.9354 | 0.9321 | 0.9335 | 0.9050 |
| Decision Tree | 0.8920 | 0.9450 | 0.9075 | 0.9090 | 0.9081 | 0.8696 |
| KNN | 0.9166 | 0.9833 | 0.9320 | 0.9271 | 0.9293 | 0.8992 |
| Naive Bayes | 0.7639 | 0.9672 | 0.7744 | 0.7694 | 0.7677 | 0.7154 |
| Support Vector Machine | 0.9221 | 0.9949 | 0.9359 | 0.9330 | 0.9343 | 0.9059 |

### Observations on Model Performance

| ML Model Name | Observation About Model Performance |
|---|---|
| Logistic Regression | Achieved an accuracy of 92.14% and an AUC score of 0.9948. It demonstrated strong classification performance across the seven classes. Its performance was very close to that of SVM, indicating that a linear classification approach is also effective for this dataset. |
| Decision Tree | Achieved an accuracy of 89.20%. It performed reasonably well but showed lower accuracy and MCC compared with Logistic Regression, KNN, and SVM. |
| KNN | Achieved an accuracy of 91.66%. Feature scaling helped the distance-based algorithm perform effectively. It demonstrated competitive precision, recall, and F1 scores. |
| Naive Bayes | Achieved an accuracy of 76.39%, which was the lowest among the five models. Although its AUC score was 0.9672, its classification performance based on accuracy, precision, recall, F1, and MCC was comparatively lower. |
| Support Vector Machine | Achieved the highest accuracy of 92.21%, AUC of 0.9949, precision of 0.9359, recall of 0.9330, F1 score of 0.9343, and MCC of 0.9059. It demonstrated the best overall performance among the five evaluated models. |

### Overall Winner for the Dataset

**Support Vector Machine (SVM)** was the best-performing model for the selected Dry Bean Dataset.

It achieved the highest accuracy of 92.21%, along with the highest AUC, precision, recall, F1 score, and MCC among the five evaluated models.

However, Logistic Regression also demonstrated very competitive performance, with an accuracy of 92.14%. The small difference between these two models indicates that both approaches are effective for this classification problem.

Based on the experimental results, SVM was selected as the overall winner.

## 5. Streamlit Web Application

An interactive web application was developed using Streamlit to demonstrate the trained classification models.

### Application Features

1. **Dataset Upload:** Allows users to upload a test dataset in CSV format containing all 16 input features and the target column.
2. **Model Selection:** Provides a dropdown to select any one of the five trained classification models.
3. **Evaluation Metrics:** Displays Accuracy, AUC, Precision, Recall, F1 Score, and MCC for the selected model.
4. **Confusion Matrix:** Displays a confusion matrix to visualize actual and predicted classifications.
5. **Classification Report:** Displays class-wise precision, recall, F1 score, and support.

The application loads the previously trained models from saved model files and evaluates the uploaded test dataset without retraining the models.

**Live Streamlit Application:** [INSERT YOUR DEPLOYED STREAMLIT APP LINK HERE]

---

**Technologies Used**

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook

