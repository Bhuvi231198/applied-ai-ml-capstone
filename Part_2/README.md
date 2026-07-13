# Part 2 – Supervised Machine Learning Model

## Project Overview

This project builds supervised machine learning models using the cleaned dataset produced in Part 1. Two predictive models were developed:

- A Regression model to predict house sale prices.
- A Binary Classification model to classify houses as having prices above or below the median sale price.

The project includes preprocessing, feature encoding, model training, evaluation, regularization experiments, and confidence interval estimation.

---

# Dataset

The cleaned dataset (`cleaned_data.csv`) generated in Part 1 is used as the input for this project.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

# Folder Structure

```
Part_2/

│
├── data/
│   └── cleaned_data.csv
│
├── outputs/
│   └── roc_curve.png
│
├── models/
│
├── src/
│   └── part2.py
│
├── notebook.ipynb
└── README.md
```

---

# Target Variables

## Regression Target

SalePrice

The regression model predicts the continuous house sale price.

## Classification Target

A binary target was created using the median SalePrice.

```
y_clf = (SalePrice > SalePrice.median()).astype(int)
```

Class meanings:

- 0 → House price less than or equal to the median
- 1 → House price greater than the median

---

# Data Preprocessing

The cleaned dataset from Part 1 was loaded.

The following preprocessing steps were performed:

- Separated feature matrix and target variables.
- One-hot encoded all categorical variables.
- Dropped the first dummy column to avoid multicollinearity.
- Split the dataset into training (80%) and testing (20%).
- Applied StandardScaler only on the training data.
- Transformed both training and testing datasets using the fitted scaler.

## Why was the scaler fitted only on the training set?

Fitting the scaler on the complete dataset would introduce **data leakage**, because statistics from the test data (mean and standard deviation) would influence the training process.

By fitting only on the training data, the test data remains completely unseen until model evaluation.

---

# Feature Encoding

The dataset contains categorical variables without a natural order.

Therefore, One-Hot Encoding was applied.

One-Hot Encoding converts each category into a separate binary column and avoids introducing artificial numerical relationships that Label Encoding would create.

The first dummy column was dropped to reduce multicollinearity.

---

# Regression Model

## Linear Regression

A Linear Regression model was trained using the scaled training dataset.

The following evaluation metrics were computed:

- Mean Squared Error (MSE)
- R² Score

The regression coefficients were extracted and ranked according to their absolute values.

The three largest coefficients indicate the features having the greatest influence on predicted SalePrice.

### Coefficient Interpretation

A large positive coefficient indicates that increasing the corresponding standardized feature increases the predicted SalePrice.

A large negative coefficient indicates that increasing the standardized feature decreases the predicted SalePrice.

---

# Ridge Regression

A Ridge Regression model was trained using:

```
alpha = 1.0
```

The Ridge model was evaluated using:

- MSE
- R² Score

Both models were compared.

## Why Ridge Regression?

Linear Regression can produce unstable coefficients when predictors are highly correlated.

Ridge Regression introduces L2 regularization, shrinking coefficient values while retaining all features.

The parameter alpha controls the strength of regularization.

Higher alpha values produce stronger coefficient shrinkage.

---

# Classification Model

A Logistic Regression classifier was trained using:

```
max_iter = 1000
```

The model predicts whether a house price is above the median.

---

# Class Balance

The binary target generated using the median produced an approximately balanced dataset.

Therefore, no imbalance handling techniques such as SMOTE or class_weight='balanced' were required.

---

# Classification Metrics

The following evaluation metrics were computed:

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- Area Under the Curve (AUC)

---

# Precision and Recall

Precision measures the proportion of predicted positive observations that are actually positive.

Formula:

```
Precision = TP / (TP + FP)
```

Recall measures the proportion of actual positive observations correctly identified.

Formula:

```
Recall = TP / (TP + FN)
```

For this dataset, Precision and Recall are both important because the classes are balanced.

The F1 Score provides a balanced summary of both metrics.

---

# ROC Curve and AUC

The ROC Curve illustrates the relationship between the True Positive Rate and False Positive Rate across different classification thresholds.

The AUC value measures the model's ability to distinguish between the two classes.

An AUC close to 1 indicates excellent class separation, while an AUC close to 0.5 indicates performance similar to random guessing.

---

# Decision Threshold Analysis

Classification probabilities were generated using:

```
predict_proba()
```

Thresholds from 0.30 to 0.70 were evaluated.

For each threshold:

- Precision
- Recall
- F1 Score

were calculated.

The threshold with the highest F1 Score provides the best balance between Precision and Recall.

Increasing the threshold generally increases Precision but reduces Recall.

Lowering the threshold generally increases Recall but decreases Precision.

---

# Logistic Regression Regularization

A second Logistic Regression model was trained using:

```
C = 0.01
```

Performance was compared against the default model (C = 1.0).

## Meaning of C

C is the inverse of regularization strength.

Smaller C values apply stronger L2 regularization.

Stronger regularization generally reduces coefficient magnitude and may improve generalization but can also reduce predictive performance if applied excessively.

---

# Bootstrap Confidence Interval

Bootstrap resampling (500 iterations) was used to estimate the confidence interval for the difference in AUC between the two Logistic Regression models.

For each bootstrap sample:

- AUC was calculated for both models.
- The AUC difference was recorded.

The following values were reported:

- Mean AUC Difference
- 2.5th Percentile
- 97.5th Percentile

If the confidence interval excludes zero, the difference between the models is considered consistent across different samples.

If the interval includes zero, the observed difference may not be statistically reliable.

---

# Output Files

The following output is generated:

- roc_curve.png

---

# How to Run

Install dependencies:

```
pip install -r requirements.txt
```

Run:

```
python Part_2/src/part2.py
```

---

# Conclusion

Both regression and classification models were successfully developed and evaluated.

Linear Regression and Ridge Regression were compared using regression metrics.

Logistic Regression was evaluated using multiple classification metrics, ROC analysis, threshold sensitivity analysis, regularization experiments, and bootstrap confidence intervals.

The trained models and evaluation results demonstrate a complete supervised machine learning workflow suitable for predictive analytics.