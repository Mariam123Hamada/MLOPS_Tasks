# Module 1 — Productionizing a Machine Learning Model

## Baseline Results

- Validation RMSE: 42.9022
- Validation MAE: 4.2994

The baseline model was developed in:
    `notebooks/00-baseline.ipynb`

This notebook intentionally represents the "before" version of the project. It contains the complete machine learning workflow in notebook form, including data loading, preprocessing, feature engineering, training, evaluation, and model serialization.

The production refactor moves these responsibilities into a reusable Python package while preserving the original modeling logic.