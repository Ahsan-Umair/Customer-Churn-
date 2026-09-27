# Telco Customer Churn

An end-to-end classification project that predicts customer churn for a telecommunications provider. It combines exploratory analysis with reusable scikit-learn pipelines, stratified cross-validation, model comparison, hyperparameter tuning, and final test evaluation.

## Workflow

- Clean the `TotalCharges` field and inspect missing values and target balance.
- Explore churn across tenure, charges, contract, internet service, support, payment method, security, and billing choices.
- Keep preprocessing inside a `ColumnTransformer` and `Pipeline` to avoid leakage.
- Compare logistic regression, decision tree, random forest, and gradient boosting.
- Use stratified F1-based validation and inspect decision-tree overfitting by depth.
- Tune decision-tree, random-forest, and logistic-regression candidates.
- Evaluate the selected pipeline with accuracy, precision, recall, F1, ROC AUC, and a confusion matrix.
- Export the full fitted pipeline.

## Notebook walkthrough

1. Inspect the Telco table and convert blank `TotalCharges` entries to numeric zero before dropping the identifier. Explore churn by tenure, charges, contract, internet and support services, payment method, security, billing, and senior-citizen status.
2. Split the raw predictors and `Churn` label into stratified training and test sets. Use a `ColumnTransformer` to scale tenure/charges and one-hot encode service categories inside each model pipeline.
3. Establish a logistic-regression baseline with five-fold stratified cross-validation scored on F1 for the `Yes` class; compare training and validation scores for overfitting.
4. Compare decision tree, random forest, and gradient boosting pipelines. Vary tree depth to expose overfitting, then tune tree, forest, and logistic-regression hyperparameters.
5. Inspect the chosen logistic model's positive and negative coefficients, and evaluate the final pipeline on the held-out test set with accuracy, precision, recall, F1, ROC AUC, and a confusion matrix.
6. Save and reload the full pipeline. It accepts raw columns in the same schema as the notebook and applies its own preprocessing.

## Dataset

`WA_Fn-UseC_-Telco-Customer-Churn.csv` contains 7,043 customer records. It covers customer demographics, services, contract and billing choices, tenure, monthly and total charges, and the `Churn` target.

## Repository contents

| File | Purpose |
| --- | --- |
| `Customer Churn.ipynb` | Complete analysis, pipeline comparison, tuning, and evaluation |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Source dataset |
| `customer_churn_pipeline.joblib` | Exported preprocessing and classification pipeline |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install jupyter numpy pandas matplotlib seaborn scikit-learn joblib
jupyter lab "Customer Churn.ipynb"
```

Use raw columns matching the training data when calling the saved pipeline; it applies its own categorical encoding and numeric scaling.
