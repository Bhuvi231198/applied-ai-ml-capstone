# Part 3 – Advanced Machine Learning: Ensemble Models, Hyperparameter Tuning & Model Deployment

## Overview

This project extends the supervised machine learning models developed in Part 2 by implementing ensemble learning methods, model comparison using cross-validation, hyperparameter tuning using GridSearchCV, feature importance analysis, feature ablation, learning curve analysis, and model serialization.

The dataset used is the cleaned House Prices dataset. A binary classification target was created by classifying houses with SalePrice greater than the median as class 1 and the remaining houses as class 0.

---

# Data Preparation

The following preprocessing steps were performed:

- Loaded cleaned_data.csv
- Created binary target variable from SalePrice
- One-hot encoded categorical variables
- Split dataset into training and testing sets (80:20)
- Standardized numerical features using StandardScaler

---

# Task 1 – Default Decision Tree

A DecisionTreeClassifier was trained using default parameters.

### Results

Training Accuracy: **YOUR VALUE**

Test Accuracy: **YOUR VALUE**

### Overfitting Discussion

The unconstrained decision tree achieved very high training accuracy but lower test accuracy, indicating overfitting.

Decision trees are considered high-variance models because they greedily choose the best split at every node without revisiting previous decisions. As the tree grows deeper, it memorizes the training data, making it sensitive to small variations and reducing its ability to generalize to unseen data.

---

# Task 2 – Controlled Decision Tree

Model parameters:

- max_depth = 5
- min_samples_split = 20

### Results

Training Accuracy: **YOUR VALUE**

Test Accuracy: **YOUR VALUE**

### Parameter Explanation

### max_depth

Limits the maximum depth of the decision tree.

A smaller depth reduces model complexity and variance but may increase bias.

### min_samples_split

Prevents a node from splitting unless it contains at least 20 samples.

This avoids creating branches based on small noisy subsets of data.

### Comparison

Compared with the default decision tree, the constrained tree generally has a smaller gap between training and testing accuracy, indicating improved generalization and reduced overfitting.

---

# Task 3 – Gini vs Entropy

Two Decision Tree models were trained using:

- criterion='gini'
- criterion='entropy'

### Results

| Criterion | Test Accuracy |
|-----------|--------------|
| Gini | **YOUR VALUE** |
| Entropy | **YOUR VALUE** |

### Gini Impurity Formula

\[
Gini = 1-\sum p_i^2
\]

### Entropy Formula

\[
Entropy = -\sum p_i \log_2(p_i)
\]

A node with Gini = 0 means every sample belongs to the same class (pure node).

---

# Task 4 – Random Forest

Model Parameters

- n_estimators = 100
- max_depth = 10
- random_state = 42

### Results

Training Accuracy: **YOUR VALUE**

Test Accuracy: **YOUR VALUE**

ROC-AUC: **YOUR VALUE**

---

## Top 5 Important Features

| Feature | Importance |
|----------|-----------|
| Feature 1 | YOUR VALUE |
| Feature 2 | YOUR VALUE |
| Feature 3 | YOUR VALUE |
| Feature 4 | YOUR VALUE |
| Feature 5 | YOUR VALUE |

### Feature Importance Explanation

Random Forest computes feature importance by measuring the average reduction in Gini impurity contributed by each feature across all trees.

Unlike linear regression coefficients, feature importance measures how useful a feature is for splitting the data rather than estimating the magnitude of its linear relationship with the target.

---

## Bagging Explanation

Random Forest uses bootstrap aggregation (bagging).

Each tree is trained on a random sample drawn with replacement from the training dataset.

Additionally, at every split only a random subset of √(number of features) is considered.

Since every tree learns slightly different patterns, averaging predictions reduces variance and produces a more stable model than a single decision tree.

---

# Task 4a – Gradient Boosting

Model Parameters

- n_estimators = 100
- learning_rate = 0.1
- max_depth = 3

### Results

Training Accuracy: **YOUR VALUE**

Test Accuracy: **YOUR VALUE**

ROC-AUC: **YOUR VALUE**

---

# Task 4b – Feature Ablation Study

The five least important features identified from Random Forest feature importance were removed.

A second Random Forest was trained using the remaining features.

### Results

Full Model ROC-AUC: **YOUR VALUE**

Reduced Model ROC-AUC: **YOUR VALUE**

### Interpretation

If both ROC-AUC values are similar, the removed features contributed very little and mainly added noise.

If ROC-AUC decreases noticeably, the removed features still contained useful predictive information.

A simpler model has lower inference cost and easier maintenance, but should only be adopted if predictive performance remains acceptable.

---

# Task 5 – Cross Validation

Five-fold Stratified Cross Validation was performed using ROC-AUC.

| Model | Mean AUC | Std AUC |
|--------|----------|----------|
| Logistic Regression | YOUR VALUE | YOUR VALUE |
| Decision Tree | YOUR VALUE | YOUR VALUE |
| Random Forest | YOUR VALUE | YOUR VALUE |
| Gradient Boosting | YOUR VALUE | YOUR VALUE |

### Why Cross Validation?

Cross-validation evaluates the model across multiple train-test splits rather than relying on a single split.

This provides a more reliable estimate of how well the model generalizes to unseen data and reduces the impact of random variation.

---

# Task 6 – GridSearchCV

Pipeline

- SimpleImputer(strategy='median')
- StandardScaler()
- RandomForestClassifier(random_state=42)

Parameter Grid

```
n_estimators = [50,100,200]

max_depth = [5,10,None]

min_samples_leaf = [1,5]
```

### Best Parameters

```
YOUR OUTPUT HERE
```

### Best Cross Validation ROC-AUC

**YOUR VALUE**

### Number of Configurations

3 × 3 × 2 = **18 parameter combinations**

Using 5-fold CV:

18 × 5 = **90 total model fits**

### Grid Search vs Random Search

Grid Search evaluates every possible parameter combination and usually finds the optimal solution but is computationally expensive.

Randomized Search evaluates only randomly selected combinations, making it faster for large parameter spaces.

---

# Task 7 – Learning Curve

| Training Fraction | Training AUC | Test AUC |
|------------------|-------------|----------|
|20%|YOUR VALUE|YOUR VALUE|
|40%|YOUR VALUE|YOUR VALUE|
|60%|YOUR VALUE|YOUR VALUE|
|80%|YOUR VALUE|YOUR VALUE|
|100%|YOUR VALUE|YOUR VALUE|

### Interpretation

Training AUC generally decreases as more training data becomes available because the model has less opportunity to memorize small datasets.

Test AUC generally increases with additional training data.

If test AUC is still increasing at 100% training size, the model is likely data-limited.

If test AUC has plateaued, model capacity rather than dataset size is the limiting factor.

---

# Task 8 – Model Serialization

The best model pipeline was saved using:

```python
joblib.dump(best_pipeline, "best_model.pkl")
```

The model was successfully reloaded using:

```python
loaded_model = joblib.load("best_model.pkl")

predictions = loaded_model.predict(sample_rows)
```

The model successfully generated predictions without errors.

---

# Final Model Comparison

| Model | 5-Fold CV Mean AUC | 5-Fold CV Std AUC | Test ROC-AUC |
|--------|-------------------|-------------------|-------------|
| Logistic Regression | YOUR VALUE | YOUR VALUE | YOUR VALUE |
| Decision Tree | YOUR VALUE | YOUR VALUE | YOUR VALUE |
| Random Forest | YOUR VALUE | YOUR VALUE | YOUR VALUE |
| Gradient Boosting | YOUR VALUE | YOUR VALUE | YOUR VALUE |

---

# Recommendation

Based on the experimental results, the Random Forest / Best GridSearch Pipeline (choose whichever achieved the highest CV AUC) is recommended for deployment.

It achieved the highest cross-validated ROC-AUC while maintaining strong performance on the independent test set. Ensemble learning reduced the variance observed in a single decision tree, resulting in improved robustness and generalization. The tuned pipeline further optimized model performance and provides a reproducible workflow suitable for deployment and future predictions.

---

# Files Included

- part3.py
- best_model.pkl
- model_comparison.csv
- README.md