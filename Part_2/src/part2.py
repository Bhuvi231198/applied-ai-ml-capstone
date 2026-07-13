# ==========================================
# Part 2 - Supervised Machine Learning
# ==========================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================
# Load Dataset
# ==========================================

current_dir = os.path.dirname(__file__)

data_path = os.path.join(
    current_dir,
    "..",
    "data",
    "cleaned_data.csv"
)

output_dir = os.path.join(
    current_dir,
    "..",
    "outputs"
)

model_dir = os.path.join(
    current_dir,
    "..",
    "models"
)

os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

df = pd.read_csv(data_path)

print(df.shape)

# ==========================================
# Target Variables
# ==========================================

# Regression Target
y_reg = df["SalePrice"]

# Binary Classification Target
y_clf = (y_reg > y_reg.median()).astype(int)

# Feature Matrix
X = df.drop(columns=["SalePrice"])

print()

print("Regression Target")

print(y_reg.head())

print()

print("Classification Target")

print(y_clf.value_counts())

# ==========================================
# Encoding
# ==========================================

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)

print()

print("Encoded Feature Shape")

print(X.shape)

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X,
    y_reg,
    test_size=0.2,
    random_state=42
)

_, _, y_clf_train, y_clf_test = train_test_split(
    X,
    y_clf,
    test_size=0.2,
    random_state=42
)

print()

print(X_train.shape)

print(X_test.shape)

# ==========================================
# Scaling
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print()

print("Scaling Complete")

# ==========================================
# Linear Regression
# ==========================================

print("\n" + "=" * 60)
print("Linear Regression")
print("=" * 60)

linear_model = LinearRegression()

linear_model.fit(X_train_scaled, y_reg_train)

y_pred_reg = linear_model.predict(X_test_scaled)

mse = mean_squared_error(y_reg_test, y_pred_reg)
r2 = r2_score(y_reg_test, y_pred_reg)

print(f"MSE : {mse:.2f}")
print(f"R2 Score : {r2:.4f}")

# ==========================================
# Feature Coefficients
# ==========================================

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": linear_model.coef_
})

coefficients["Absolute"] = coefficients["Coefficient"].abs()

coefficients = coefficients.sort_values(
    by="Absolute",
    ascending=False
)

print("\nTop 10 Features by Absolute Coefficient")

print(coefficients.head(10))

print("\nTop 3 Most Important Features")

print(coefficients.head(3)[["Feature", "Coefficient"]])

# ==========================================
# Ridge Regression
# ==========================================

print("\n" + "=" * 60)
print("Ridge Regression")
print("=" * 60)

ridge_model = Ridge(alpha=1.0)

ridge_model.fit(X_train_scaled, y_reg_train)

ridge_predictions = ridge_model.predict(X_test_scaled)

ridge_mse = mean_squared_error(
    y_reg_test,
    ridge_predictions
)

ridge_r2 = r2_score(
    y_reg_test,
    ridge_predictions
)

print(f"MSE : {ridge_mse:.2f}")
print(f"R2 Score : {ridge_r2:.4f}")

# ==========================================
# Ridge vs Linear Comparison
# ==========================================

comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Ridge Regression"
    ],
    "MSE": [
        mse,
        ridge_mse
    ],
    "R2 Score": [
        r2,
        ridge_r2
    ]
})

print("\nRegression Comparison")

print(comparison)

# ==========================================
# Classification - Check Class Balance
# ==========================================

print("\n" + "=" * 60)
print("Class Distribution")
print("=" * 60)

print(y_clf_train.value_counts())

class_percentage = y_clf_train.value_counts(normalize=True) * 100

print("\nClass Percentage")

print(class_percentage)

# ==========================================
# Logistic Regression
# ==========================================

print("\n" + "=" * 60)
print("Logistic Regression")
print("=" * 60)

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train_scaled, y_clf_train)

y_pred_clf = log_model.predict(X_test_scaled)

y_prob = log_model.predict_proba(X_test_scaled)[:, 1]

# ==========================================
# Classification Metrics
# ==========================================

cm = confusion_matrix(y_clf_test, y_pred_clf)

print("\nConfusion Matrix")

print(cm)

print("\nClassification Report")

print(classification_report(y_clf_test, y_pred_clf))

auc = roc_auc_score(y_clf_test, y_prob)

print(f"\nAUC Score : {auc:.4f}")

# ==========================================
# ROC Curve
# ==========================================

fpr, tpr, thresholds = roc_curve(
    y_clf_test,
    y_prob
)

plt.figure(figsize=(7,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc:.3f}"
)

plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "roc_curve.png"
    )
)

plt.show()

# ==========================================
# Threshold Analysis
# ==========================================

threshold_results = []

for threshold in np.arange(0.30, 0.71, 0.10):

    prediction = (y_prob >= threshold).astype(int)

    precision = precision_score(
        y_clf_test,
        prediction
    )

    recall = recall_score(
        y_clf_test,
        prediction
    )

    f1 = f1_score(
        y_clf_test,
        prediction
    )

    threshold_results.append({
        "Threshold": round(threshold,2),
        "Precision": round(precision,3),
        "Recall": round(recall,3),
        "F1": round(f1,3)
    })

threshold_df = pd.DataFrame(threshold_results)

print("\nThreshold Comparison")

print(threshold_df)

# ==========================================
# Logistic Regression (Strong Regularization)
# ==========================================

print("\n" + "=" * 60)
print("Logistic Regression (C = 0.01)")
print("=" * 60)

log_model_c001 = LogisticRegression(
    C=0.01,
    max_iter=1000
)

log_model_c001.fit(
    X_train_scaled,
    y_clf_train
)

y_pred_c001 = log_model_c001.predict(X_test_scaled)

y_prob_c001 = log_model_c001.predict_proba(
    X_test_scaled
)[:,1]

precision_c001 = precision_score(
    y_clf_test,
    y_pred_c001
)

recall_c001 = recall_score(
    y_clf_test,
    y_pred_c001
)

auc_c001 = roc_auc_score(
    y_clf_test,
    y_prob_c001
)

comparison_lr = pd.DataFrame({
    "Model":[
        "Logistic (C=1.0)",
        "Logistic (C=0.01)"
    ],
    "Precision":[
        precision_score(y_clf_test, y_pred_clf),
        precision_c001
    ],
    "Recall":[
        recall_score(y_clf_test, y_pred_clf),
        recall_c001
    ],
    "AUC":[
        auc,
        auc_c001
    ]
})

print("\nRegularization Comparison")

print(comparison_lr)

# ==========================================
# Bootstrap Confidence Interval
# ==========================================

print("\n" + "=" * 60)
print("Bootstrap Confidence Interval")
print("=" * 60)

np.random.seed(42)

auc_differences = []

y_true = np.array(y_clf_test)

prob1 = np.array(y_prob)

prob2 = np.array(y_prob_c001)

for _ in range(500):

    indices = np.random.choice(
        len(y_true),
        size=len(y_true),
        replace=True
    )

    y_sample = y_true[indices]

    p1 = prob1[indices]

    p2 = prob2[indices]

    # Skip samples containing only one class
    if len(np.unique(y_sample)) < 2:
        continue

    auc1 = roc_auc_score(
        y_sample,
        p1
    )

    auc2 = roc_auc_score(
        y_sample,
        p2
    )

    auc_differences.append(
        auc1 - auc2
    )

mean_difference = np.mean(auc_differences)

lower = np.percentile(
    auc_differences,
    2.5
)

upper = np.percentile(
    auc_differences,
    97.5
)

print(f"Mean Difference : {mean_difference:.4f}")

print(f"95% CI : ({lower:.4f}, {upper:.4f})")

if lower > 0 or upper < 0:
    print("Confidence interval excludes zero.")
else:
    print("Confidence interval includes zero.")