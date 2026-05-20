# RetainX — AI-Powered Customer Retention Intelligence Platform

RetainX is an AI-driven customer churn prediction and retention intelligence platform developed for the banking sector. The platform combines machine learning, explainable AI, customer analytics, revenue-risk estimation, and strategic retention simulation to help organizations proactively identify churn-prone customers and take data-driven retention decisions.

---

# Problem Statement

Customer churn is one of the biggest hidden revenue leaks in the banking industry.

Traditional churn systems usually:

* only predict whether a customer may leave,
* provide little explainability,
* lack business context,
* and fail to recommend actionable retention strategies.

RetainX transforms churn prediction into a complete business decision-support system.

---

# Objectives

* Predict customer churn accurately using machine learning
* Identify high-value customers at risk
* Estimate potential revenue loss
* Prioritize customers requiring immediate action
* Recommend personalized retention strategies
* Simulate retention campaigns and estimate ROI

---

# Key Features

### AI-Powered Churn Prediction

Implemented and compared multiple machine learning models:

* Logistic Regression
* Random Forest
* XGBoost (Final Production Model)

### Advanced Feature Engineering

Engineered business-driven features including:

* Customer Value Score
* Revenue at Risk
* Digital Engagement Score
* Balance-to-Salary Ratio
* Financial Stress Indicators
* Churn Risk Score
* Product Engagement Metrics

### Explainable AI (SHAP)

Integrated SHAP explainability for transparent model interpretation:

* SHAP Feature Importance
* SHAP Beeswarm Analysis
* Individual Customer Waterfall Explanations

This helps explain:

* why customers churn,
* major churn drivers,
* and feature-level impact on predictions.

### Customer Prioritization Engine

Built a prioritization framework to identify:

* Critical churners
* High-value customers
* Revenue-sensitive customers
* Recoverable revenue opportunities

### AI Retention Recommendation Engine

Developed a rule-based recommendation system suggesting:

* Relationship manager allocation
* Loyalty reward campaigns
* Cross-sell opportunities
* Wealth management upgrades
* Digital re-engagement campaigns

based on customer behavior and churn probability.

### What-If Strategy Simulator

Designed a strategic simulation module where managers can simulate:

* Retention budgets
* Cashback offers
* Loyalty incentives
* Campaign strength

The system estimates:

* Expected churn reduction
* Revenue saved
* ROI impact

This converts predictive AI into actionable strategic AI.

---

# Tech Stack

### Programming Language

* Python

### Libraries & Frameworks

* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP
* Plotly
* Matplotlib
* Streamlit
* Imbalanced-learn (SMOTE)
* Joblib

---

# Project Structure

RetainX/

* data/

  * raw/
  * processed/
  * eda_plots/
  * shap_outputs/

* models/

* Final_notebook.ipynb

* app.py

* README.md

* requirements.txt

---

# Machine Learning Pipeline

### 1. Data Cleaning

* Missing value handling
* Duplicate removal
* Outlier treatment
* Data standardization

### 2. Feature Engineering

* Behavioral indicators
* Financial indicators
* Engagement indicators
* Business-driven churn metrics

### 3. Preprocessing

* Label Encoding
* Feature Scaling
* SMOTE balancing
* Train-test splitting

### 4. Model Training

* Logistic Regression
* Random Forest
* XGBoost

### 5. Model Evaluation

Evaluation metrics used:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

### 6. Explainable AI

Integrated SHAP for:

* global model interpretation,
* feature impact analysis,
* customer-level explainability.

### 7. Business Intelligence Layer

* Revenue risk analysis
* Customer prioritization
* Retention recommendation system
* Strategic retention simulation

---

# Business Impact

RetainX enables organizations to:

* Reduce customer churn proactively
* Prioritize high-value customers
* Minimize revenue leakage
* Improve retention ROI
* Support strategic decision-making
* Increase customer lifetime value

---

# Outputs Generated

### Processed Datasets

* Cleaned datasets
* Feature-engineered datasets
* Customer prioritization outputs
* Retention recommendation outputs

### Visual Analytics

* EDA visualizations
* SHAP explainability plots

### Saved ML Models

* Logistic Regression
* Random Forest
* XGBoost production model

---

# Future Enhancements

* Deep Learning integration
* Real-time API deployment
* Cloud deployment
* Real-time streaming analytics
* Automated campaign optimization
* LLM-powered conversational analytics

---

# How to Run

### Clone Repository

git clone [https://github.com/your-username/RetainX.git](https://github.com/your-username/RetainX.git)

cd RetainX

### Install Dependencies

pip install -r requirements.txt

### Run Jupyter Notebook

Open:
Final_notebook.ipynb

and execute cells sequentially.

### Run Streamlit Dashboard

streamlit run app.py

---

# Conclusion

RetainX goes beyond traditional churn prediction systems by combining machine learning with explainable AI, customer intelligence, revenue-risk analysis, and strategic retention planning.

The platform enables organizations to transform predictive insights into actionable business decisions.

---

# License

This project is intended for educational, research, and hackathon purposes.
