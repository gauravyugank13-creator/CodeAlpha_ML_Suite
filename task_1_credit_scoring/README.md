# Task 1: Credit Scoring Model

This project aims to predict the creditworthiness of loan applicants by evaluating structured historical financial data. It classifies applicants as either low-risk or high-risk (likely to default).

## Project Objective
Develop a robust, production-ready machine learning classification pipeline. The model learns from historical applicant attributes (e.g., credit history, debt levels, income stability, employment duration) to output risk probabilities and binary default predictions.

## Folder Structure
The project follows a modular, clean structure designed for reproducibility and easy integration:

```text
task_1_credit_scoring/
├── README.md                  # Project overview and instructions
├── requirements_task1.txt     # Task-specific python dependencies
├── data/
│   ├── raw/                   # Raw input dataset files (CSV, Parquet, etc.)
│   └── processed/             # Cleaned, encoded, and scaled dataset splits
├── notebooks/
│   └── exploration.ipynb      # Jupyter notebook for exploratory data analysis (EDA)
├── src/
│   ├── __init__.py            # Package indicator
│   ├── config.py              # Central config for paths, split params, features, seeds
│   ├── data_loader.py         # Handles dataset loading and initial checks
│   ├── preprocessing.py       # Handles imputation, scaling, and categorical encoding
│   ├── feature_engineering.py  # Calculates domain-specific financial features
│   ├── model_builder.py       # Classifiers (Logistic Regression, Random Forest, etc.)
│   ├── trainer.py             # Coordinates fitting, hyperparameters, checkpoints
│   ├── evaluator.py           # Calculates metrics (ROC-AUC, confusion matrix, report)
│   └── predictor.py           # Runs batch and single-record predictions (inference)
├── models/
│   └── .gitkeep               # Saved serialized model checkpoints (.joblib)
├── results/
│   ├── plots/                 # Metrics visualization (ROC curves, feature importance)
│   ├── metrics/               # Evaluation results text and JSON files
│   └── predictions/           # Generated batch inference output files
└── app/
    └── placeholder.md         # Deployment plan documentation
```

## Expected Workflow
1. **Exploratory Data Analysis (EDA):** Identify distributions, check for missing values, analyze target imbalance, and find correlation patterns.
2. **Feature Engineering & Preprocessing:** Develop features (e.g. debt-to-income, credit utilization ratio), impute missing values, scale numerical columns, and encode categorical labels.
3. **Model Selection & Tuning:** Train baseline classifiers (Logistic Regression, Decision Tree, Random Forest) and tune hyperparameters using cross-validation.
4. **Evaluation:** Produce validation reports containing ROC-AUC curves, confusion matrices, precision/recall curves, and feature importance analyses.
5. **Deployment Preparation:** Build inference helper scripts and wrap predictions inside a serving pipeline (e.g., Flask UI or API).

## Future Execution Plan
1. **Phase 1 (Current):** Setup skeletal framework, configurations, placeholder modules, and documentation.
2. **Phase 2:** Dataset acquisition, schema verification, data validation, and exploratory data analysis.
3. **Phase 3:** Feature engineering pipeline and preprocessing implementation.
4. **Phase 4:** Model training, evaluation, comparison, and selection.
5. **Phase 5:** Deployment to local Flask serving interface.
