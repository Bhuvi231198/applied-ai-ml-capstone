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

