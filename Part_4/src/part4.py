import os
import pickle
import pandas as pd
from transformers import pipeline


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_model.pkl"
)

TEST_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "test.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "predictions.csv"
)

REPORT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "ai_report.txt"
)

# Create output folder
os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("Loading trained model...")

    with open(
        MODEL_PATH,
        "rb"
    ) as file:

        model = pickle.load(file)

    print("Model loaded successfully")

    return model



# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading test data...")

    df = pd.read_csv(
        TEST_DATA_PATH
    )

    print(
        "Data shape:",
        df.shape
    )

    return df



# ============================================================
# PREPARE INPUT FEATURES
# ============================================================

def prepare_input(df):

    print("Preparing input data...")


    X = df.copy()


    # Remove target column if it exists
    possible_targets = [
        "target",
        "label",
        "price",
        "SalePrice"
    ]


    for col in possible_targets:

        if col in X.columns:

            X = X.drop(
                columns=[col]
            )


    return X



# ============================================================
# PREDICTION
# ============================================================

def predict(model, X):

    print("Generating predictions...")

    predictions = model.predict(
        X
    )

    return predictions



# ============================================================
# SAVE OUTPUT
# ============================================================

def save_output(df, predictions):

    print("Saving prediction file...")


    result = df.copy()

    result["prediction"] = predictions


    result.to_csv(
        OUTPUT_PATH,
        index=False
    )


    print(
        "Saved:",
        OUTPUT_PATH
    )

def load_ai_model():

    print("Loading AI explanation model...")


    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-small"
    )


    print("AI model loaded successfully")


    return generator


# ============================================================
# GENERATE EXPLANATION
# ============================================================

def generate_explanation(
        generator,
        prediction,
        important_features
):


    prompt = f"""
    Explain this machine learning prediction.

    Prediction:
    {prediction}

    Important features:
    {important_features}

    Explain in simple language why this prediction happened.
    """


    result = generator(
        prompt,
        max_length=150
    )


    return result[0]["generated_text"]

# ============================================================
# GET FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(model, X):

    print("Extracting feature importance...")


    importance_data = {}


    if hasattr(model, "feature_importances_"):

        importance_values = model.feature_importances_

        for feature, value in zip(
            X.columns,
            importance_values
        ):

            importance_data[feature] = value


    elif hasattr(model, "coef_"):

        coefficients = model.coef_

        for feature, value in zip(
            X.columns,
            coefficients
        ):

            importance_data[feature] = abs(value)


    else:

        print(
            "Feature importance not available for this model"
        )

        return {}


    sorted_features = dict(
        sorted(
            importance_data.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )


    return sorted_features

# ============================================================
# SAVE AI REPORT
# ============================================================

def save_ai_report(
        data,
        predictions,
        explanations
):

    print("Saving AI report...")


    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:


        file.write(
            "AI Prediction Report\n"
        )

        file.write(
            "=" * 50 + "\n\n"
        )


        for i in range(len(predictions)):


            file.write(
                f"Sample {i+1}\n"
            )

            file.write(
                "-" * 30 + "\n"
            )


            file.write(
                f"Prediction: {predictions[i]}\n\n"
            )


            file.write(
                "AI Explanation:\n"
            )

            file.write(
                explanations[i]
            )

            file.write(
                "\n\n"
            )


    print(
        "Report saved:",
        REPORT_PATH
    )

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":


    model = load_model()


    data = load_data()


    X = prepare_input(
        data
    )


    predictions = predict(
        model,
        X
    )
    feature_importance = get_feature_importance(
        model,
        X
    )


    top_features = dict(
        list(feature_importance.items())[:5]
    )    


    save_output(
        data,
        predictions
    )


    # -------------------------------
    # AI Explanation
    # -------------------------------

    ai_model = load_ai_model()


    print("\nGenerating AI explanations...")


    explanations = []


for prediction in predictions[:5]:


    explanation = generate_explanation(
    ai_model,
    prediction,
    top_features
    )


    explanations.append(
        explanation
    )


    print("\nPrediction:")
    print(prediction)

    print("Explanation:")
    print(explanation)



save_ai_report(
    data.head(5),
    predictions[:5],
    explanations
)






