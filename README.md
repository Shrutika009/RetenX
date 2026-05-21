# RetainX

RetainX is a Streamlit-based customer retention intelligence app for banking churn analysis. It predicts customer churn risk, estimates revenue at risk, prioritizes customers for retention teams, and suggests targeted retention actions.

## Live Demo

Streamlit Cloud URL:

```text
https://retainx.streamlit.app
```

Use this after the app is deployed and the `retainx` subdomain is available.

## Features

- Churn prediction using Logistic Regression, Random Forest, and XGBoost
- Customer risk tiers from low risk to critical
- Revenue-at-risk and recoverable-revenue estimates
- Customer prioritization queue for retention teams
- Rule-based retention recommendations
- SHAP explainability and model feature importance views
- What-if simulator for campaign budget and retention strength

## Tech Stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Plotly
- Joblib

## Machine Learning Pipeline

1. Cleaned and prepared banking customer data.
2. Engineered churn-focused behavioral, financial, and engagement features.
3. Encoded categorical variables and scaled model inputs.
4. Trained and compared Logistic Regression, Random Forest, and XGBoost models.
5. Selected XGBoost as the production model for churn probability scoring.
6. Generated SHAP explainability outputs for feature-level model interpretation.
7. Built a business layer for customer prioritization, retention actions, and campaign simulation.

## Business Impact

- Helps identify customers most likely to churn before revenue is lost
- Prioritizes high-value and high-risk customers for retention teams
- Estimates revenue at risk and recoverable revenue opportunity
- Recommends targeted retention actions based on customer behavior
- Supports data-driven campaign planning through what-if simulation
- Improves explainability by showing the main drivers behind churn risk

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Project Structure

```text
RetainX/
├── app.py
├── requirements.txt
├── runtime.txt
├── models/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── eda_plots/
│   └── shap_outputs/
└── Bank_Churn_Final_notebook.ipynb
```

## Model Artifacts

The trained models, encoders, scaler, processed datasets, and SHAP outputs are included in the repository so the Streamlit app can run directly after deployment.
