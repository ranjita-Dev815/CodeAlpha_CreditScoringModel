"""
Task 1: Credit Scoring Model
Dataset: Kaggle "Give Me Some Credit" -> cs-training.csv
Run:  python credit_scoring.py
Install: pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix,
                             classification_report)
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv("cs-training.csv", index_col=0)
df = df.rename(columns={"SeriousDlqin2yrs": "target"})   # 1 = defaulter
print("Shape:", df.shape)
print(df["target"].value_counts(normalize=True))          # imbalance check

# ---------------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------------
df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)
df = df[(df["age"] > 18) & (df["age"] < 100)]
df = df[df["RevolvingUtilizationOfUnsecuredLines"] <= 5]  # remove outliers
df = df[df["DebtRatio"] <= df["DebtRatio"].quantile(0.99)]

# ---------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------
df["TotalLateCount"] = (df["NumberOfTime30-59DaysPastDueNotWorse"]
                        + df["NumberOfTime60-89DaysPastDueNotWorse"]
                        + df["NumberOfTimes90DaysLate"])
df["IncomePerPerson"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
df["DebtToIncome"] = df["DebtRatio"] * df["MonthlyIncome"]
df["HasLatePayment"] = (df["TotalLateCount"] > 0).astype(int)

# ---------------------------------------------------------------
# 4. SPLIT + SCALE + SMOTE (SMOTE sirf train data par)
# ---------------------------------------------------------------
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train_s, y_train)

# ---------------------------------------------------------------
# 5. TRAIN MODELS
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10,
                                            n_jobs=-1, random_state=42),
}

results, probs = [], {}
for name, model in models.items():
    model.fit(X_train_bal, y_train_bal)
    pred = model.predict(X_test_s)
    prob = model.predict_proba(X_test_s)[:, 1]
    probs[name] = prob
    results.append({
        "Model": name,
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-Score": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, prob),
    })

res_df = pd.DataFrame(results).round(4)
print("\n=== MODEL COMPARISON ===")
print(res_df.to_string(index=False))
res_df.to_csv("model_results.csv", index=False)

# ---------------------------------------------------------------
# 6. PLOTS
# ---------------------------------------------------------------
# ROC curves
plt.figure(figsize=(7, 5))
for name, prob in probs.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, prob):.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves"); plt.legend(); plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150); plt.close()

# Confusion matrix of best model (by ROC-AUC)
best = res_df.sort_values("ROC-AUC", ascending=False).iloc[0]["Model"]
best_pred = models[best].predict(X_test_s)
sns.heatmap(confusion_matrix(y_test, best_pred), annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix - {best}")
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150); plt.close()

# Feature importance (Random Forest)
imp = pd.Series(models["Random Forest"].feature_importances_,
                index=X.columns).sort_values()
imp.plot(kind="barh", figsize=(7, 5), title="Feature Importance (Random Forest)")
plt.tight_layout(); plt.savefig("feature_importance.png", dpi=150); plt.close()

print(f"\nBest model: {best}")
print(classification_report(y_test, best_pred))
print("Saved: model_results.csv, roc_curves.png, confusion_matrix.png, feature_importance.png")
