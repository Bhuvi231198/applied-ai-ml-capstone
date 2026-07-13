# ==========================================
# Part 1 - Data Acquisition & Exploratory Analysis
# Task 1 - Load and Inspect Dataset
# ==========================================

import os
import pandas as pd

# -----------------------------
# Load Dataset
# -----------------------------

# Build path relative to this script
current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, "..", "data", "train.csv")

# Read CSV
df = pd.read_csv(data_path)

# -----------------------------
# Display Basic Information
# -----------------------------

print("=" * 60)
print("First Five Rows")
print("=" * 60)
print(df.head())

print("\n")

print("=" * 60)
print("Data Types")
print("=" * 60)
print(df.dtypes)

print("\n")

print("=" * 60)
print("Dataset Shape")
print("=" * 60)
print(df.shape)

print("\n")

print("=" * 60)
print("Column Names")
print("=" * 60)
print(df.columns.tolist())