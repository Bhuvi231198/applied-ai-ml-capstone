# ==========================================
# Part 1 - Data Acquisition & Exploratory Analysis
# Task 1 - Load and Inspect Dataset
# ==========================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Current folder
current_dir = os.path.dirname(__file__)

# Output folder
output_dir = os.path.join(current_dir, "..", "outputs")
os.makedirs(output_dir, exist_ok=True)

# Better plot appearance
sns.set_theme(style="whitegrid")

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

# ==========================================
# Task 3 - Duplicate Detection & Removal
# ==========================================

print("\n" + "=" * 60)
print("Duplicate Detection")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print(f"Number of duplicate rows before removal: {duplicate_count}")

# Store null percentages before removing duplicates
null_percentage_before = (df.isnull().sum() / len(df)) * 100

# Remove duplicates
df = df.drop_duplicates()

print(f"Dataset shape after removing duplicates: {df.shape}")

duplicate_count_after = df.duplicated().sum()

print(f"Number of duplicate rows after removal: {duplicate_count_after}")

# Compare null percentages
null_percentage_after = (df.isnull().sum() / len(df)) * 100

null_change = pd.DataFrame({
    "Before (%)": null_percentage_before,
    "After (%)": null_percentage_after
})

print("\nNull Percentage Comparison")
print(null_change)

# ==========================================
# Task 4 - Data Type Correction
# ==========================================

print("\n" + "=" * 60)
print("Data Type Correction")
print("=" * 60)

memory_before = df.memory_usage(deep=True).sum()

print(f"Memory Usage Before: {memory_before:,} bytes")

# Convert MSSubClass to category
df["MSSubClass"] = df["MSSubClass"].astype("category")

# Convert Neighborhood to category
df["Neighborhood"] = df["Neighborhood"].astype("category")

memory_after = df.memory_usage(deep=True).sum()

print(f"Memory Usage After : {memory_after:,} bytes")

print("\nUpdated Data Types")

print(df[["MSSubClass", "Neighborhood"]].dtypes)

# ==========================================
# Task 5 - Descriptive Statistics & Skewness
# ==========================================

print("\n" + "=" * 60)
print("Descriptive Statistics")
print("=" * 60)

# Numeric columns only
numeric_df = df.select_dtypes(include=["number"])

print(numeric_df.describe())

# -----------------------------------------
# Calculate Skewness
# -----------------------------------------

print("\n" + "=" * 60)
print("Skewness of Numeric Columns")
print("=" * 60)

skewness = numeric_df.skew()

# Create DataFrame
skewness_df = pd.DataFrame({
    "Column": skewness.index,
    "Skewness": skewness.values,
    "Absolute Skewness": skewness.abs().values
})

# Sort by absolute skewness
skewness_df = skewness_df.sort_values(
    by="Absolute Skewness",
    ascending=False
)

print(skewness_df)

# Most skewed column
most_skewed_column = skewness_df.iloc[0]["Column"]
most_skewed_value = skewness_df.iloc[0]["Skewness"]

print("\n" + "=" * 60)
print("Most Skewed Column")
print("=" * 60)

print(f"Column : {most_skewed_column}")
print(f"Skewness : {most_skewed_value:.3f}")

# Save for later tasks
top_two_skewed = skewness_df.head(2)["Column"].tolist()

# ==========================================
# Task 6 - Outlier Detection using IQR
# ==========================================

print("\n" + "=" * 60)
print("Outlier Detection using IQR")
print("=" * 60)

# Choose two important numeric columns
iqr_columns = ["SalePrice", "LotArea"]

outlier_summary = []

for col in iqr_columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    outlier_count = df[
        (df[col] < lower_bound) |
        (df[col] > upper_bound)
    ].shape[0]

    outlier_summary.append({
        "Column": col,
        "Q1": round(Q1, 2),
        "Q3": round(Q3, 2),
        "IQR": round(IQR, 2),
        "Lower Bound": round(lower_bound, 2),
        "Upper Bound": round(upper_bound, 2),
        "Outlier Count": outlier_count
    })

outlier_df = pd.DataFrame(outlier_summary)

print(outlier_df)

# ==========================================
# Task 7.1 - Line Plot
# ==========================================

plt.figure(figsize=(10,5))

plt.plot(df.index, df["SalePrice"])

plt.title("Sale Price Across Dataset")
plt.xlabel("Row Index")
plt.ylabel("Sale Price")

plt.tight_layout()

plt.savefig(os.path.join(output_dir, "line_plot_saleprice.png"))

plt.show()

# ==========================================
# Task 7.2 - Bar Chart
# ==========================================

plt.figure(figsize=(12,6))

bar_data = df.groupby("Neighborhood")["SalePrice"].mean().sort_values(ascending=False)

plt.bar(bar_data.index, bar_data.values)

plt.xticks(rotation=90)

plt.title("Average Sale Price by Neighborhood")
plt.xlabel("Neighborhood")
plt.ylabel("Average Sale Price")

plt.tight_layout()

plt.savefig(os.path.join(output_dir, "bar_chart_neighborhood.png"))

plt.show()

# ==========================================
# Task 7.3 - Histogram
# ==========================================

plt.figure(figsize=(8,5))

sns.histplot(
    df[most_skewed_column],
    bins=20,
    kde=True
)

plt.title(f"Histogram of {most_skewed_column}")
plt.xlabel(most_skewed_column)
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "histogram_most_skewed.png"
    )
)

plt.show()

# ==========================================
# Task 7.4 - Scatter Plot
# ==========================================

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="GrLivArea",
    y="SalePrice"
)

plt.title("Ground Living Area vs Sale Price")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "scatter_saleprice.png"
    )
)

plt.show()

# ==========================================
# Task 7.5 - Box Plot
# ==========================================

plt.figure(figsize=(12,6))

sns.boxplot(
    data=df,
    x="OverallQual",
    y="SalePrice"
)

plt.title("Sale Price by Overall Quality")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "boxplot_saleprice.png"
    )
)

plt.show()

# ==========================================
# Task 8 - Correlation Heatmap
# ==========================================

numeric_df = df.select_dtypes(include=["number"])

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(18,14))

sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    annot=False
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "correlation_heatmap.png"
    )
)

plt.show()

# -----------------------------------------
# Highest Correlated Pair
# -----------------------------------------

corr = correlation_matrix.abs()

upper_triangle = corr.where(
    ~np.tril(np.ones(corr.shape), k=0).astype(bool)
)

highest_pair = upper_triangle.stack().idxmax()

highest_value = upper_triangle.stack().max()

print("\nHighest Correlated Pair")

print(highest_pair)

print(f"Correlation : {highest_value:.3f}")