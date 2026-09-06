# Customer Churn Prediction
## Live Demo

[Open the Customer Churn Prediction App](https://churn-prediction-acj4eno9lhbodkge4ftrhp.streamlit.app/)

An end-to-end Machine Learning classification project to predict customer churn using Python and Scikit-learn.

The project focuses on improving churn detection by handling class imbalance, comparing classification models, and optimizing the decision threshold.

## Dataset

The dataset contains 10,000 customer records with features such as:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card Ownership
- Active Membership
- Estimated Salary

Target variable: `Exited`

- `0` - Customer stayed
- `1` - Customer churned

## Project Workflow

- Data cleaning and preprocessing
- Exploratory Data Analysis
- Train-test split with stratification
- One-Hot Encoding
- Feature Scaling
- Logistic Regression baseline
- Random Forest classification
- Class imbalance handling
- Threshold optimization
- Model evaluation

## Model Performance

### Random Forest

- Accuracy: 86.05%
- Precision: 77.59%
- Recall: 44.23%
- F1 Score: 56.34%
- ROC-AUC: 0.8508

### Threshold Optimized Model

Selected threshold: **0.33**

- Accuracy: 83.55%
- Precision: 59.65%
- Recall: 59.21%
- F1 Score: 59.43%
- ROC-AUC: 0.8508

Threshold optimization improved the model's ability to identify churned customers while maintaining a reasonable balance between precision and recall.

## Tech Stack

Python | Pandas | NumPy | Scikit-learn | Matplotlib | Jupyter Notebook

## Key Learning

This project demonstrates why accuracy alone is not enough for an imbalanced classification problem.

By focusing on recall, precision, F1-score, ROC-AUC, and threshold tuning, the model provides a more practical approach to identifying customers at risk of churn.
