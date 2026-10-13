# CodeAlpha_CreditScoringMode
Machine Learning Internship Task 1 at **CodeAlpha**: predict an individual's creditworthiness from past financial data.

## Objective
Build a classification model that predicts whether a person is likely to face serious financial delinquency (default) within 2 years.

## Dataset
Kaggle: [Give Me Some Credit (2011 Competition Data)](https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset)
Download `cs-training.csv` and keep it in the same folder as `credit_scoring.py`.

- 150,000 records, 10 features
- Highly imbalanced: only about 6.7% are defaulters

## Approach
1. **Data cleaning:** filled missing `MonthlyIncome` (median) and `NumberOfDependents` (0), removed unrealistic ages and extreme outliers.
2. **Feature engineering:** `TotalLateCount`, `IncomePerPerson`, `DebtToIncome`, `HasLatePayment`.
3. **Scaling:** StandardScaler.
4. **Imbalance handling:** SMOTE applied on the training set only.
5. **Models:** Logistic Regression, Decision Tree, Random Forest.
6. **Evaluation:** Precision, Recall, F1-Score, ROC-AUC.

## Results

| Model | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.1968 | 0.7640 | 0.3130 | 0.8524 |
| Decision Tree | 0.2359 | 0.6387 | 0.3445 | 0.8233 |
| **Random Forest** | **0.2734** | 0.6261 | **0.3806** | **0.8535** |

**Best model: Random Forest** (highest F1 and ROC-AUC).

Accuracy alone is misleading here because about 93% of customers are non-defaulters, so Recall, F1 and ROC-AUC are the main metrics.

## Output files
- `model_results.csv`: metrics comparison
- `roc_curves.png`: ROC curves of all models
- `confusion_matrix.png`: confusion matrix of the best model
- `feature_importance.png`: Random Forest feature importance

## How to run
```bash
pip install -r requirements.txt
python credit_scoring.py
```
Or upload the files to Google Colab and run `!python credit_scoring.py`.

## Tech stack
Python, Pandas, NumPy, Scikit-learn, Imbalanced-learn, Matplotlib, Seaborn

## Author
Your Name | CodeAlpha Machine Learning Intern
