# Cyber Traffic Intrusion Detection System

#### This repository presents a machine learning–based intrusion detection system for classifying network traffic as Normal or Attack.
It combines a deployed end-to-end implementation with a rigorous model evaluation workflow.

### Project Overview

Intrusion detection requires balancing missed attacks (false negatives) against false alarms (false positives).
This project demonstrates both practical deployment and evidence-based model selection for this task.

### The deployed Streamlit application demonstrates:

End-to-end preprocessing and inference
Real-time prediction capability
Practical ML system design

A simplified dataset is used to ensure lightweight inference and usability.
The focus of this component is system implementation rather than exhaustive benchmarking.

## Extended Evaluation (Analysis Scope)

A separate experimental analysis was conducted using the UNSW-NB15 dataset to rigorously evaluate model performance.

Task framed as binary classification (Normal vs Attack)

Predefined train–test split respected to avoid data leakage

Separate preprocessing for numerical and categorical features using pipelines

Models and Evaluation

### Models evaluated:

○ Dummy Classifier

○ Logistic Regression

○ Class-weighted Logistic Regression

○ Random Forest

### Evaluation metrics:

○ Precision, Recall, F1-score

○ Confusion Matrix

○ ROC-AUC

○ Precision–Recall Average Precision

## Key Results

Final model: Logistic Regression

ROC-AUC: 0.87

Average Precision: 0.90

Logistic Regression outperformed Random Forest due to the high-dimensional sparse feature space created by one-hot encoding.
Random Forest resulted in higher false negatives and was therefore rejected.

Evaluation Notebook

All experiments, analyses, and visualizations are available in:
├──notebooks/intrusion_detection_evaluation.ipynb

Tech Stack
├──Python
├──Pandas, NumPy
├──Scikit-learn
├──Streamlit
├──Matplotlib, Seaborn

Key Takeaway

This project emphasizes evidence-based model selection and demonstrates that simpler models can outperform complex ensembles when aligned with data characteristics and problem constraints.
