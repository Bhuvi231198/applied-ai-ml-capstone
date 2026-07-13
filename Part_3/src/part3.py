# ==========================================
# Part 3 - Ensemble Models, Tuning & Pipeline
# ==========================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score
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

print("=" * 60)
print("Dataset")
print("=" * 60)

print(df.shape)

# ==========================================
# Target Variables
# ==========================================

y_reg = df["SalePrice"]

y_clf = (
    y_reg >
    y_reg.median()
).astype(int)

X = df.drop(
    columns=["SalePrice"]
)

# ==========================================
# One-Hot Encoding
# ==========================================

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)

print("\nEncoded Shape")

print(X.shape)

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_clf_train, y_clf_test = train_test_split(
    X,
    y_clf,
    test_size=0.20,
    random_state=42,
    stratify=y_clf
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ==========================================
# Feature Scaling
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\nScaling Complete")

# ==========================================
# Logistic Regression Baseline
# ==========================================

print("\n" + "=" * 60)
print("Logistic Regression Baseline")
print("=" * 60)

log_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

log_model.fit(
    X_train_scaled,
    y_clf_train
)

log_train_pred = log_model.predict(
    X_train_scaled
)

log_test_pred = log_model.predict(
    X_test_scaled
)

log_test_prob = log_model.predict_proba(
    X_test_scaled
)[:,1]

log_train_acc = accuracy_score(
    y_clf_train,
    log_train_pred
)

log_test_acc = accuracy_score(
    y_clf_test,
    log_test_pred
)

log_auc = roc_auc_score(
    y_clf_test,
    log_test_prob
)

print(f"Training Accuracy : {log_train_acc:.4f}")
print(f"Test Accuracy     : {log_test_acc:.4f}")
print(f"ROC AUC           : {log_auc:.4f}")

# ==========================================
# Task 1
# Unconstrained Decision Tree
# ==========================================

print("\n" + "=" * 60)
print("Task 1 - Default Decision Tree")
print("=" * 60)

dt_default = DecisionTreeClassifier(
    random_state=42
)

dt_default.fit(
    X_train_scaled,
    y_clf_train
)

dt_train_pred = dt_default.predict(
    X_train_scaled
)

dt_test_pred = dt_default.predict(
    X_test_scaled
)

dt_train_acc = accuracy_score(
    y_clf_train,
    dt_train_pred
)

dt_test_acc = accuracy_score(
    y_clf_test,
    dt_test_pred
)

print(f"Training Accuracy : {dt_train_acc:.4f}")
print(f"Test Accuracy     : {dt_test_acc:.4f}")

print()

if dt_train_acc - dt_test_acc > 0.10:
    print("Observation : Model appears to be overfitting.")
else:
    print("Observation : No strong evidence of overfitting.")

# ==========================================
# Task 2
# Controlled Decision Tree
# ==========================================

print("\n" + "=" * 60)
print("Task 2 - Controlled Decision Tree")
print("=" * 60)

dt_controlled = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    random_state=42
)

dt_controlled.fit(
    X_train_scaled,
    y_clf_train
)

dt_control_train_pred = dt_controlled.predict(
    X_train_scaled
)

dt_control_test_pred = dt_controlled.predict(
    X_test_scaled
)

dt_control_train_acc = accuracy_score(
    y_clf_train,
    dt_control_train_pred
)

dt_control_test_acc = accuracy_score(
    y_clf_test,
    dt_control_test_pred
)

print(f"Training Accuracy : {dt_control_train_acc:.4f}")
print(f"Test Accuracy     : {dt_control_test_acc:.4f}")

print("\nComparison")

comparison = pd.DataFrame({

    "Model":[
        "Default Decision Tree",
        "Controlled Decision Tree"
    ],

    "Training Accuracy":[
        round(dt_train_acc,4),
        round(dt_control_train_acc,4)
    ],

    "Test Accuracy":[
        round(dt_test_acc,4),
        round(dt_control_test_acc,4)
    ]

})

print(comparison)

# ==========================================
# Task 3
# Gini vs Entropy Comparison
# ==========================================

print("\n" + "=" * 60)
print("Task 3 - Gini vs Entropy")
print("=" * 60)

dt_gini = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

dt_entropy = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)

dt_gini.fit(
    X_train_scaled,
    y_clf_train
)

dt_entropy.fit(
    X_train_scaled,
    y_clf_train
)

gini_pred = dt_gini.predict(
    X_test_scaled
)

entropy_pred = dt_entropy.predict(
    X_test_scaled
)

gini_accuracy = accuracy_score(
    y_clf_test,
    gini_pred
)

entropy_accuracy = accuracy_score(
    y_clf_test,
    entropy_pred
)

print(f"Gini Test Accuracy    : {gini_accuracy:.4f}")
print(f"Entropy Test Accuracy : {entropy_accuracy:.4f}")

# ==========================================
# Task 4
# Random Forest
# ==========================================

print("\n" + "=" * 60)
print("Task 4 - Random Forest")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

rf_model.fit(
    X_train_scaled,
    y_clf_train
)

rf_train_pred = rf_model.predict(
    X_train_scaled
)

rf_test_pred = rf_model.predict(
    X_test_scaled
)

rf_train_prob = rf_model.predict_proba(
    X_train_scaled
)[:,1]

rf_test_prob = rf_model.predict_proba(
    X_test_scaled
)[:,1]

rf_train_accuracy = accuracy_score(
    y_clf_train,
    rf_train_pred
)

rf_test_accuracy = accuracy_score(
    y_clf_test,
    rf_test_pred
)

rf_auc = roc_auc_score(
    y_clf_test,
    rf_test_prob
)

print(f"Training Accuracy : {rf_train_accuracy:.4f}")
print(f"Test Accuracy     : {rf_test_accuracy:.4f}")
print(f"ROC-AUC           : {rf_auc:.4f}")

# ==========================================
# Feature Importance
# ==========================================

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": rf_model.feature_importances_

})

feature_importance = feature_importance.sort_values(

    by="Importance",

    ascending=False

)

print("\nTop 5 Important Features")

print(feature_importance.head(5))

top5_features = feature_importance.head(5)

lowest5_features = feature_importance.tail(5)

print("\nLowest 5 Features")

print(lowest5_features)

print("\nRandom Forest Summary")

rf_summary = pd.DataFrame({

    "Metric":[

        "Training Accuracy",

        "Test Accuracy",

        "ROC-AUC"

    ],

    "Value":[

        round(rf_train_accuracy,4),

        round(rf_test_accuracy,4),

        round(rf_auc,4)

    ]

})

print(rf_summary)

# ==========================================
# Task 4a
# Gradient Boosting Classifier
# ==========================================

print("\n" + "=" * 60)
print("Task 4a - Gradient Boosting")
print("=" * 60)

gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb_model.fit(
    X_train_scaled,
    y_clf_train
)

gb_train_pred = gb_model.predict(X_train_scaled)

gb_test_pred = gb_model.predict(X_test_scaled)

gb_train_prob = gb_model.predict_proba(
    X_train_scaled
)[:, 1]

gb_test_prob = gb_model.predict_proba(
    X_test_scaled
)[:, 1]

gb_train_accuracy = accuracy_score(
    y_clf_train,
    gb_train_pred
)

gb_test_accuracy = accuracy_score(
    y_clf_test,
    gb_test_pred
)

gb_auc = roc_auc_score(
    y_clf_test,
    gb_test_prob
)

print(f"Training Accuracy : {gb_train_accuracy:.4f}")
print(f"Test Accuracy     : {gb_test_accuracy:.4f}")
print(f"ROC-AUC           : {gb_auc:.4f}")

# ==========================================
# Task 4b
# Feature Ablation Study
# ==========================================

print("\n" + "=" * 60)
print("Task 4b - Feature Ablation")
print("=" * 60)

# Five least important feature names
lowest_feature_names = lowest5_features["Feature"].tolist()

print("\nRemoving Features:")

for feature in lowest_feature_names:
    print(feature)

# Remove them from train and test

X_train_reduced = X_train.drop(
    columns=lowest_feature_names
)

X_test_reduced = X_test.drop(
    columns=lowest_feature_names
)

# Scale reduced dataset

scaler_reduced = StandardScaler()

X_train_reduced_scaled = scaler_reduced.fit_transform(
    X_train_reduced
)

X_test_reduced_scaled = scaler_reduced.transform(
    X_test_reduced
)

# Train another Random Forest

rf_reduced = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

rf_reduced.fit(
    X_train_reduced_scaled,
    y_clf_train
)

rf_reduced_prob = rf_reduced.predict_proba(
    X_test_reduced_scaled
)[:,1]

rf_reduced_auc = roc_auc_score(
    y_clf_test,
    rf_reduced_prob
)

print("\nAUC Comparison")

print(f"Full Model AUC    : {rf_auc:.4f}")

print(f"Reduced Model AUC : {rf_reduced_auc:.4f}")

# ==========================================
# Ablation Summary
# ==========================================

ablation_results = pd.DataFrame({

    "Model":[
        "Random Forest (All Features)",
        "Random Forest (Without Lowest 5)"
    ],

    "ROC-AUC":[
        round(rf_auc,4),
        round(rf_reduced_auc,4)
    ]

})

print("\nFeature Ablation Summary")

print(ablation_results)

difference = rf_auc - rf_reduced_auc

print(f"\nAUC Difference : {difference:.4f}")

if abs(difference) < 0.01:
    print("Observation : Removing the lowest-importance features had very little impact on performance.")

elif difference > 0:
    print("Observation : The removed features contributed to the model performance.")

else:
    print("Observation : Removing the features slightly improved the model, suggesting they mainly added noise.")

# ==========================================
# Task 5
# Cross Validation Comparison
# ==========================================

print("\n" + "=" * 60)
print("Task 5 - 5-Fold Cross Validation")
print("=" * 60)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=20,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )

}

cv_results = []

for model_name, model in models.items():

    pipeline = make_pipeline(

        StandardScaler(),

        model

    )

    scores = cross_val_score(

        pipeline,

        X,

        y_clf,

        cv=cv,

        scoring="roc_auc",

        n_jobs=-1

    )

    cv_results.append({

        "Model": model_name,

        "Mean AUC": scores.mean(),

        "Std AUC": scores.std()

    })

cv_results_df = pd.DataFrame(cv_results)

cv_results_df["Mean AUC"] = cv_results_df["Mean AUC"].round(4)

cv_results_df["Std AUC"] = cv_results_df["Std AUC"].round(4)

print("\n5-Fold Cross Validation Results\n")

print(cv_results_df.sort_values(
    by="Mean AUC",
    ascending=False
))

# ==========================================
# Best Cross Validated Model
# ==========================================

best_cv_model = cv_results_df.sort_values(

    by="Mean AUC",

    ascending=False

).iloc[0]

print("\nBest Model Based on Cross Validation\n")

print(best_cv_model)

# ==========================================
# Ranking
# ==========================================

ranking = cv_results_df.sort_values(

    by="Mean AUC",

    ascending=False

).reset_index(drop=True)

ranking.index = ranking.index + 1

print("\nModel Ranking\n")

print(ranking)

# ==========================================
# Task 6
# GridSearchCV with Pipeline
# ==========================================

print("\n" + "=" * 60)
print("Task 6 - GridSearchCV")
print("=" * 60)

pipeline = make_pipeline(

    SimpleImputer(strategy="median"),

    StandardScaler(),

    RandomForestClassifier(
        random_state=42
    )

)

param_grid = {

    "randomforestclassifier__n_estimators": [
        50,
        100,
        200
    ],

    "randomforestclassifier__max_depth": [
        5,
        10,
        None
    ],

    "randomforestclassifier__min_samples_leaf": [
        1,
        5
    ]

}

grid_search = GridSearchCV(

    estimator=pipeline,

    param_grid=param_grid,

    cv=StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    ),

    scoring="roc_auc",

    n_jobs=-1,

    verbose=1

)

grid_search.fit(

    X_train,

    y_clf_train

)

print("\nBest Parameters\n")

print(grid_search.best_params_)

print("\nBest Cross Validation ROC-AUC")

print(round(grid_search.best_score_,4))

best_pipeline = grid_search.best_estimator_

print("\nBest Pipeline\n")

print(best_pipeline)

# ==========================================
# Total Configurations
# ==========================================

num_models = (

    len(param_grid["randomforestclassifier__n_estimators"])

    *

    len(param_grid["randomforestclassifier__max_depth"])

    *

    len(param_grid["randomforestclassifier__min_samples_leaf"])

)

total_fits = num_models * 5

print("\nGrid Search Statistics")

print(f"Parameter combinations : {num_models}")

print(f"Total model fits (5-fold CV) : {total_fits}")

# ==========================================
# Evaluate Best Pipeline
# ==========================================

best_predictions = best_pipeline.predict(
    X_test
)

best_probabilities = best_pipeline.predict_proba(
    X_test
)[:,1]

best_accuracy = accuracy_score(
    y_clf_test,
    best_predictions
)

best_auc = roc_auc_score(
    y_clf_test,
    best_probabilities
)

print("\nBest Pipeline Performance")

print(f"Test Accuracy : {best_accuracy:.4f}")

print(f"Test ROC-AUC  : {best_auc:.4f}")

grid_summary = pd.DataFrame({

    "Metric":[

        "Best CV ROC-AUC",

        "Test Accuracy",

        "Test ROC-AUC"

    ],

    "Value":[

        round(grid_search.best_score_,4),

        round(best_accuracy,4),

        round(best_auc,4)

    ]

})

print("\nGrid Search Summary")

print(grid_summary)

# ==========================================
# Task 7
# Manual Learning Curve
# ==========================================

print("\n" + "=" * 60)
print("Task 7 - Manual Learning Curve")
print("=" * 60)

fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

learning_curve_results = []

for fraction in fractions:

    rows = int(len(X_train) * fraction)

    X_subset = X_train.iloc[:rows]

    y_subset = y_clf_train.iloc[:rows]

    best_pipeline.fit(
        X_subset,
        y_subset
    )

    train_prob = best_pipeline.predict_proba(
        X_subset
    )[:,1]

    test_prob = best_pipeline.predict_proba(
        X_test
    )[:,1]

    train_auc = roc_auc_score(
        y_subset,
        train_prob
    )

    test_auc = roc_auc_score(
        y_clf_test,
        test_prob
    )

    learning_curve_results.append({

        "Training Fraction": f"{int(fraction*100)}%",

        "Training AUC": round(train_auc,4),

        "Test AUC": round(test_auc,4)

    })

learning_curve_df = pd.DataFrame(
    learning_curve_results
)

print()

print(learning_curve_df)

# ==========================================
# Task 8
# Save Best Model
# ==========================================

print("\n" + "=" * 60)
print("Saving Best Model")
print("=" * 60)

model_path = os.path.join(
    model_dir,
    "best_model.pkl"
)

joblib.dump(
    best_pipeline,
    model_path
)

print(f"Model saved to:\n{model_path}")

# ==========================================
# Reload Saved Model
# ==========================================

loaded_model = joblib.load(
    model_path
)

print("\nModel Reloaded Successfully")

# ==========================================
# Predict on Two Sample Rows
# ==========================================

sample_rows = X_test.iloc[:2]

predictions = loaded_model.predict(
    sample_rows
)

probabilities = loaded_model.predict_proba(
    sample_rows
)

print("\nPredictions")

print(predictions)

print("\nPrediction Probabilities")

print(probabilities)

# ==========================================
# Final Model Comparison
# ==========================================

print("\n" + "=" * 60)
print("Final Model Comparison")
print("=" * 60)

final_results = pd.DataFrame({

    "Model":[

        "Logistic Regression",

        "Decision Tree",

        "Random Forest",

        "Gradient Boosting",

        "Best GridSearch Pipeline"

    ],

    "Test Accuracy":[

        round(log_test_acc,4),

        round(dt_control_test_acc,4),

        round(rf_test_accuracy,4),

        round(gb_test_accuracy,4),

        round(best_accuracy,4)

    ],

    "Test ROC-AUC":[

        round(log_auc,4),

        round(
            roc_auc_score(
                y_clf_test,
                dt_controlled.predict_proba(X_test_scaled)[:,1]
            ),
            4
        ),

        round(rf_auc,4),

        round(gb_auc,4),

        round(best_auc,4)

    ]

})

print(final_results)

comparison_path = os.path.join(
    output_dir,
    "model_comparison.csv"
)

final_results.to_csv(
    comparison_path,
    index=False
)

print("\nComparison table saved.")

print("\n" + "=" * 60)
print("PART 3 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("✔ Decision Tree")
print("✔ Controlled Decision Tree")
print("✔ Gini vs Entropy")
print("✔ Random Forest")
print("✔ Feature Importance")
print("✔ Gradient Boosting")
print("✔ Feature Ablation")
print("✔ Cross Validation")
print("✔ GridSearchCV")
print("✔ Learning Curve")
print("✔ Model Serialization")
print("✔ Model Reload")
print("✔ Final Comparison")

