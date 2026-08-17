# Machine Learning Assignment 2 - Bank Marketing Classification

## a. Problem Statement

The objective of this project is to build and compare multiple machine learning
classification models for predicting whether a bank client will subscribe to a
term deposit. The work covers dataset preparation, model training, evaluation,
comparison, and deployment through an interactive Streamlit web application.

## b. Dataset Description

Dataset used: UCI Bank Marketing Dataset

Source: https://archive.ics.uci.edu/dataset/222/bank+marketing

The dataset contains information collected from direct marketing campaigns of a
Portuguese banking institution. The classification target is `y`, which indicates
whether the customer subscribed to a term deposit.

- Number of instances used: 41,188
- Number of input features: 20
- Target variable: `y`
- Target classes: `yes`, `no`
- Problem type: Binary classification
- Feature types: Numeric and categorical

The dataset satisfies the assignment requirement because it has more than 500
instances and more than 12 features.

## c. GitHub Repository Link

Add your GitHub repository link here after pushing the project:

`https://github.com/<your-username>/<your-repository-name>`

## Live Streamlit App Link

Add your deployed Streamlit Community Cloud link here:

`https://<your-app-name>.streamlit.app`

## d. Models Used

The following five classification models were implemented as required in the
assignment.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8645 | 0.9431 | 0.9263 | 0.8645 | 0.8827 | 0.5792 |
| Decision Tree | 0.8476 | 0.9413 | 0.9239 | 0.8476 | 0.8698 | 0.5555 |
| kNN | 0.9076 | 0.8963 | 0.8958 | 0.9076 | 0.8983 | 0.4640 |
| Naive Bayes | 0.8223 | 0.8427 | 0.8871 | 0.8223 | 0.8453 | 0.4028 |
| Random Forest | 0.8479 | 0.9484 | 0.9283 | 0.8479 | 0.8706 | 0.5709 |

## Model Performance Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression performs strongly on this dataset, with high AUC and good MCC. This suggests that many useful decision patterns can be captured through a linear boundary after encoding categorical features. |
| Decision Tree | The Decision Tree gives competitive AUC but lower accuracy than kNN and Logistic Regression. Its performance is useful for interpretability, but a single tree can be sensitive to splits and may not generalize as well as ensemble methods. |
| kNN | kNN achieves the highest accuracy and F1 score on the test set. However, its AUC and MCC are lower than Logistic Regression and Random Forest, which means it is less balanced in ranking positive subscription cases. |
| Naive Bayes | Naive Bayes has the weakest overall performance. This is expected because the model assumes feature independence, which is unlikely in marketing and customer behavior data. |
| Random Forest | Random Forest produces the highest AUC and the best precision among the tested models. This indicates strong ability to rank likely subscribers and handle nonlinear interactions between campaign, customer, and economic features. |
| Overall Winner for this dataset | Random Forest is selected as the overall winner because it has the highest AUC score and strong precision/MCC. Although kNN has higher accuracy, Random Forest is more reliable for this imbalanced business classification problem where ranking likely subscribers is important. |

## Repository Structure

```text
project-folder/
  app.py
  requirements.txt
  README.md
  test_data.csv
  data/
    bank-additional-full.csv
  model/
    train_models.py
  outputs/
    model_comparison.csv
    evaluation_details.json
```

## How to Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the models:

```bash
python model/train_models.py
```

Run the Streamlit app:

```bash
streamlit run app.py
```

Upload `test_data.csv` in the Streamlit app to view predictions, evaluation
metrics, confusion matrix, and classification report.

Note: On Streamlit Community Cloud, the app trains the five models from
`data/bank-additional-full.csv` if pre-saved model files are not present. This
keeps the GitHub repository lightweight and avoids large file upload limits.

## Streamlit App Features

- CSV upload option for test data
- Model selection dropdown
- Evaluation metrics display
- Confusion matrix visualization
- Classification report
- Prediction preview
- Download option for predictions

## Academic Integrity Note

This project uses the public UCI Bank Marketing dataset and a custom project
structure, preprocessing workflow, evaluation script, and Streamlit interface.
The README observations are based on the generated model metrics for this
specific experiment.
