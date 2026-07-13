# Part 1 – Data Acquisition, Cleaning and Exploratory Data Analysis

## Project Overview

This project performs data acquisition, data cleaning, exploratory data analysis (EDA), and statistical analysis on the House Prices dataset. The objective is to understand the quality of the dataset, identify missing values, detect outliers, explore relationships among variables, and prepare a cleaned dataset for machine learning in Part 2.

---

# Dataset

**Dataset Name:** House Prices - Advanced Regression Techniques

The dataset contains residential house information including property characteristics and sale prices.

## Reason for Choosing the Dataset

This dataset was selected because it satisfies all assignment requirements:

- More than 500 records
- More than 5 columns
- Contains both numeric and categorical features
- Contains missing values
- Contains outliers
- Suitable for regression problems
- Widely used for machine learning practice

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

# Folder Structure

```
part1_data_exploration/

│
├── cleaned_data.csv
├── README.md
├── notebook.ipynb
│
├── data/
│   └── train.csv
│
├── outputs/
│   ├── line_plot_saleprice.png
│   ├── bar_chart_neighborhood.png
│   ├── histogram_most_skewed.png
│   ├── scatter_saleprice.png
│   ├── boxplot_saleprice.png
│   └── correlation_heatmap.png
│
└── src/
    └── part1.py
```

---

# Steps Performed

## 1. Data Loading

- Loaded the dataset using `pd.read_csv()`.
- Displayed the first five rows.
- Printed data types.
- Printed dataset dimensions.

---

## 2. Missing Value Analysis

Missing values were calculated for every column.

Columns with more than 20% missing values were identified separately.

Numeric columns having less than 20% missing values were imputed using the median.

### Why Median?

Median is less sensitive to extreme values than the mean.

Many variables in this dataset contain skewed distributions and outliers. Using the mean would shift imputed values toward extreme observations, whereas the median better represents the typical value.

---

## 3. Duplicate Detection

Duplicate rows were identified using `duplicated()`.

Duplicates were removed using `drop_duplicates()`.

Null percentages before and after duplicate removal were compared.

---

## 4. Data Type Correction

The following conversions were performed:

- MSSubClass → Category
- Neighborhood → Category

Converting repetitive categorical values reduced memory usage and better represents their meaning.

---

## 5. Descriptive Statistics

Descriptive statistics were generated for every numeric variable using `describe()`.

Skewness was calculated for every numeric feature.

### Interpretation of Skewness

Positive skew indicates a long right tail caused by a few unusually high values.

Negative skew indicates a long left tail caused by a few unusually low values.

Because skewed distributions pull the mean toward extreme values, the median provides a more reliable measure of central tendency for missing value imputation.

---

## 6. Outlier Detection

Outliers were detected using the Interquartile Range (IQR) method.

The following variables were analysed:

- SalePrice
- LotArea

### SalePrice

Several expensive houses were detected as outliers.

These appear to represent genuine luxury properties rather than incorrect observations.

Therefore, they were retained.

### LotArea

Several unusually large land parcels were identified.

These observations were also retained because they represent valid properties.

Outliers may be transformed or handled using robust machine learning techniques during Part 2 rather than removed.

---

# Visualizations

## Line Plot

Shows how SalePrice changes across observations.

---

## Bar Chart

Compares average SalePrice across Neighborhood categories.

This helps identify neighbourhoods with higher average house prices.

---

## Histogram

The histogram of the most skewed variable demonstrates a long-tailed distribution.

This supports using the median instead of the mean for missing value imputation.

---

## Scatter Plot

Ground Living Area and SalePrice show a clear positive relationship.

Larger houses generally sell for higher prices, although some variability exists.

---

## Box Plot

SalePrice was compared across Overall Quality categories.

Higher quality houses generally have higher median prices and wider price variation.

---

# Correlation Heatmap

Pearson correlation was calculated for all numeric variables.

The strongest correlated variable pair was identified.

Correlation alone does not imply causation.

A third variable such as house quality, size, or location may influence both variables simultaneously.

---

# Mean vs Median Comparison

The two most skewed variables were compared using both mean and median.

Because both variables exhibit substantial skewness, median was selected for final missing value imputation.

After imputation, no remaining missing values existed in those columns.

---

# Spearman Correlation

Pearson and Spearman correlation matrices were computed.

The three variable pairs with the greatest differences were identified.

Spearman correlation is preferred when variables follow monotonic but non-linear relationships because it is based on ranks rather than raw values.

Pearson correlation remains appropriate for approximately linear relationships.

---

# Group Aggregation

Neighborhood was grouped and SalePrice statistics were calculated.

Mean, standard deviation, and observation count were computed.

The ratio between the highest and lowest average SalePrice demonstrates that Neighborhood contains useful predictive information.

Large within-group variation suggests that Neighborhood alone is insufficient for accurate prediction and should be combined with additional features.

---

# Output Files

The following files are generated automatically:

- cleaned_data.csv
- line_plot_saleprice.png
- bar_chart_neighborhood.png
- histogram_most_skewed.png
- scatter_saleprice.png
- boxplot_saleprice.png
- correlation_heatmap.png

---

# How to Run

Install dependencies:

```
pip install -r requirements.txt
```

Run the script:

```
python part1_data_exploration/src/part1.py
```

The cleaned dataset and plots will be generated automatically.

---

# Conclusion

The dataset was successfully cleaned, analysed, and prepared for machine learning.

The resulting `cleaned_data.csv` will be used in Part 2 for regression model development.