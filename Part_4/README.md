# Part 4 – Model Deployment and AI Explanation

## Objective

The objective of Part 4 is to load the trained machine learning model developed in Part 3, generate predictions on unseen test data, and provide easy-to-understand AI-generated explanations for those predictions.

## Features

* Load the trained machine learning model (`best_model.pkl`).
* Read test data from a CSV file.
* Prepare input features for prediction.
* Generate predictions using the trained model.
* Save prediction results to a CSV file.
* Generate natural language explanations using a free local language model.
* Extract feature importance (when supported by the trained model).
* Create a human-readable AI prediction report.

## Project Structure

```text
Part_4/
│
├── data/
│   └── test.csv
│
├── models/
│   └── best_model.pkl
│
├── outputs/
│   ├── predictions.csv
│   └── ai_report.txt
│
├── src/
│   └── part4.py
│
└── README.md
```

## Prerequisites

Install the required Python packages:

```bash
pip install pandas scikit-learn transformers torch sentencepiece
```

## How to Run

Navigate to the Part 4 directory and execute:

```bash
python src/part4.py
```

## Output

After successful execution, the following files will be generated in the `outputs` folder:

* **predictions.csv** – Contains the original test data along with model predictions.
* **ai_report.txt** – Contains prediction summaries and AI-generated explanations.

## Workflow

1. Load the trained machine learning model.
2. Load the test dataset.
3. Prepare input features.
4. Generate predictions.
5. Save prediction results.
6. Extract feature importance (if available).
7. Generate AI-based explanations.
8. Save the final AI report.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Transformers (Hugging Face)
* PyTorch

## Learning Outcomes

This part demonstrates:

* Model deployment using a saved machine learning model.
* Prediction on unseen data.
* Explainable AI using a local language model.
* Automated generation of prediction reports.
* End-to-end machine learning workflow integration.
