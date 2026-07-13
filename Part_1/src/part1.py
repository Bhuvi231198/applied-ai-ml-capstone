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

# ==========================================
# Task 2 - Missing Value Analysis
# ==========================================

print("=" * 60)
print("Missing Value Analysis")
print("=" * 60)

# Count missing values
missing_count = df.isnull().sum()

# Percentage of missing values
missing_percentage = (missing_count / len(df)) * 100

# Create summary table
missing_summary = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing Percentage": missing_percentage
})

# Sort in descending order
missing_summary = missing_summary.sort_values(
    by="Missing Percentage",
    ascending=False
)

print(missing_summary)

print("\n")

# -----------------------------------------
# Columns having more than 20% missing values
# -----------------------------------------

high_null_columns = missing_summary[
    missing_summary["Missing Percentage"] > 20
]

print("=" * 60)
print("Columns with More Than 20% Missing Values")
print("=" * 60)

if high_null_columns.empty:
    print("No columns exceed 20% missing values.")
else:
    print(high_null_columns)

print("\n")

# -----------------------------------------
# Fill numeric columns having <20% null values
# using median
# -----------------------------------------

numeric_columns = df.select_dtypes(include=["number"]).columns

for col in numeric_columns:

    null_percentage = (df[col].isnull().sum() / len(df)) * 100

    if 0 < null_percentage < 20:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

print("=" * 60)
print("Remaining Missing Values After Median Imputation")
print("=" * 60)

print(df.isnull().sum())