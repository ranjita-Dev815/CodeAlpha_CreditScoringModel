# CodeAlpha Credit Scoring Model

A machine learning project that predicts the creditworthiness of an individual from past financial data. The model estimates the probability that a person will experience serious financial delinquency (default), using the "Give Me Some Credit" dataset.

## Project Files

| File | Description |
|------|-------------|
| `credit_scoring.ipynb` | Main notebook: data cleaning, model training and evaluation |
| `model_results` | Summary of model performance metrics |
| `feature_importance` | Plot of the most influential features |
| `confusion_matrix` | Confusion matrix of the final model |
| `roc_curves` | ROC curves comparing the models |
| `requirements.txt` | Python dependencies |

## Approach

1. Load and explore the dataset
2. Handle missing values and outliers
3. Split data into training and test sets
4. Train classification models (e.g. Logistic Regression, Decision Tree, Random Forest)
5. Evaluate with accuracy, precision, recall, F1-score and ROC-AUC
6. Analyse feature importance

## Setup

```bash
pip install -r requirements.txt
jupyter notebook credit_scoring.ipynb
```

## Results

See `model_results`, `confusion_matrix`, `roc_curves` and `feature_importance` for the final evaluation outputs.

## Author

Ranjita Kumari — CodeAlpha Internship Project
