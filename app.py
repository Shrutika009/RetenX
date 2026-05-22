from pathlib import Path
from html import escape

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "processed"
SHAP_DIR = BASE_DIR / "data" / "shap_outputs"
MODEL_DIR = BASE_DIR / "models"

RISK_COLORS = {
    "Low Risk": "#426f98",
    "Medium Risk": "#c47c00",
    "High Risk": "#d63c31",
    "Critical": "#8f1737",
}


def theme_colors() -> dict[str, str]:
    return {
        "bg": "#eff6ff",
        "surface": "#ffffff",
        "soft": "#eef6ff",
        "text": "#12211d",
        "muted": "#68736f",
        "border": "#dbeafe",
        "sidebar": "#fbfdff",
        "accent": "#2563eb",
        "accent_hover": "#1d4ed8",
        "accent_soft": "#e8f1ff",
        "mint": "#05b983",
        "mint_soft": "#e4f8ef",
        "gold": "#f0c84b",
        "gold_soft": "#fff8da",
        "rose": "#e9799a",
        "rose_soft": "#fff0f5",
        "table_header": "#f8fbff",
        "table_bg": "#ffffff",
        "table_alt": "#f5f9ff",
        "table_text": "#12211d",
        "table_muted": "#74807b",
        "progress_track": "#dbeafe",
        "chart_text": "#32413c",
        "chart_grid": "#d7e7fb",
        "shadow": "rgba(30,64,175,0.10)",
    }


st.set_page_config(
    page_title="RetainX",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_theme() -> None:
    colors = theme_colors()
    st.session_state["theme_colors"] = colors
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@400;500;600;700;800;900&display=swap');

        html, body, body *, [class*="css"] {{
            font-family: "Raleway", Arial, sans-serif;
        }}

        .material-icons, .material-icons-rounded, .material-symbols-rounded,
        .material-symbols-outlined, .material-symbols-sharp {{
            font-family: "Material Symbols Rounded", "Material Icons" !important;
            font-weight: normal !important;
            font-style: normal !important;
            line-height: 1 !important;
            letter-spacing: normal !important;
            text-transform: none !important;
            white-space: nowrap !important;
            word-wrap: normal !important;
            direction: ltr !important;
            font-feature-settings: "liga" !important;
            -webkit-font-feature-settings: "liga" !important;
        }}

        .stApp, [data-testid="stAppViewContainer"] {{
            background:
                radial-gradient(circle at 7% 8%, rgba(37, 99, 235, 0.13), transparent 28%),
                radial-gradient(circle at 92% 2%, rgba(5, 185, 131, 0.13), transparent 30%),
                linear-gradient(135deg, #f8fbff 0%, {colors["bg"]} 48%, #f4fbff 100%);
            color: {colors["text"]};
        }}

        .block-container {{
            padding-top: 1.45rem;
            padding-bottom: 3rem;
            max-width: 1460px;
        }}

        [data-testid="stHeader"], [data-testid="stToolbar"], header {{
            background: transparent !important;
            color: {colors["text"]} !important;
        }}

        [data-testid="stDecoration"] {{
            display: none;
        }}

        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapseButton"] *,
        [data-testid="collapsedControl"],
        [data-testid="collapsedControl"] *,
        button[aria-label="Close sidebar"],
        button[aria-label="Close sidebar"] *,
        button[aria-label="Open sidebar"],
        button[aria-label="Open sidebar"] *,
        button[title="Close sidebar"],
        button[title="Close sidebar"] *,
        button[title="Open sidebar"] {{
            display: none !important;
            visibility: hidden !important;
        }}

        [data-testid="stSidebar"] button:first-of-type,
        [data-testid="stSidebar"] button:first-of-type * {{
            display: none !important;
            visibility: hidden !important;
            width: 0 !important;
            height: 0 !important;
            overflow: hidden !important;
        }}

        [data-testid="stSidebar"] {{
            background: rgba(255, 254, 253, 0.92);
            border-right: 1px solid {colors["border"]};
            box-shadow: 18px 0 45px rgba(15,46,110,0.06);
        }}

        [data-testid="stSidebar"] * {{
            color: {colors["text"]};
        }}

        [data-testid="stSidebar"] [role="radiogroup"] label {{
            padding: 0.66rem 0.78rem;
            border-radius: 12px;
            margin-bottom: 0.26rem;
            border: 1px solid transparent;
            color: {colors["muted"]} !important;
            font-weight: 700;
            transition: all 160ms ease;
        }}

        [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
            background: {colors["soft"]};
            border-color: {colors["border"]};
            transform: translateX(2px);
        }}

        [data-testid="stSidebar"] hr {{
            border-color: {colors["border"]};
        }}

        .sidebar-brand {{
            font-family: "Raleway", Arial, sans-serif;
            font-size: 1.8rem;
            line-height: 1.1;
            font-weight: 900;
            margin: 0.25rem 0 0.45rem 0;
            display: flex;
            align-items: center;
            letter-spacing: -0.02em;
        }}

        .sidebar-brand:before {{
            display: none;
        }}

        [role="radio"][aria-checked="true"] div:first-child {{
            background: {colors["accent"]} !important;
            border-color: {colors["accent"]} !important;
        }}

        [data-testid="stSidebar"] label:has([role="radio"][aria-checked="true"]) {{
            background: linear-gradient(135deg, {colors["accent_soft"]}, #f7fbf5);
            border-color: rgba(37,99,235,0.24);
            color: {colors["accent"]} !important;
            box-shadow: inset 4px 0 0 {colors["accent"]};
        }}

        .hero {{
            position: relative;
            padding: 1.35rem 1.45rem 1.55rem 1.45rem;
            margin-bottom: 1.1rem;
            border: 1px solid rgba(235,232,242,0.86);
            border-radius: 24px;
            background:
                linear-gradient(135deg, rgba(255,255,255,0.94), rgba(255,255,255,0.78)),
                linear-gradient(135deg, rgba(37,99,235,0.10), rgba(5,185,131,0.09));
            box-shadow: 0 24px 65px {colors["shadow"]};
            overflow: hidden;
        }}

        .hero:after {{
            display: none;
        }}

        .hero:before {{
            display: none;
        }}

        .eyebrow {{
            color: {colors["accent"]};
            font-size: 1.12rem;
            font-weight: 900;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }}

        .eyebrow:empty {{
            display: none;
        }}

        .hero h1 {{
            color: {colors["text"]};
            font-size: clamp(2rem, 3vw, 3.05rem);
            line-height: 1.12;
            letter-spacing: 0;
            margin: 0;
            font-weight: 900;
            max-width: 980px;
        }}

        .hero p {{
            color: {colors["muted"]};
            max-width: 820px;
            font-size: 1.05rem;
            margin-top: 0.65rem;
        }}

        .form-shell {{
            max-width: 980px;
            margin: 1.2rem auto 1.4rem auto;
        }}

        div[data-testid="stForm"] {{
            background:
                linear-gradient(180deg, rgba(255,255,255,0.98), rgba(255,255,255,0.91)),
                linear-gradient(135deg, rgba(37,99,235,0.12), rgba(5,185,131,0.08));
            border: 1px solid {colors["border"]};
            border-radius: 24px;
            padding: 1.2rem 1.2rem 1.35rem 1.2rem;
            box-shadow: 0 24px 65px {colors["shadow"]};
        }}

        .form-title {{
            color: {colors["text"]};
            font-size: 1.18rem;
            font-weight: 900;
            margin: 0 0 1.1rem 0;
            letter-spacing: -0.01em;
            display: flex;
            align-items: center;
        }}

        .form-title:before {{
            display: none;
        }}

        .required-label {{
            color: {colors["text"]};
            font-size: 0.9rem;
            font-weight: 800;
            margin: 0.2rem 0 0.35rem 0;
        }}

        .required-label .required-star {{
            color: #d93025;
            font-weight: 900;
            margin-left: 0.18rem;
        }}

        .field-error {{
            color: #d93025;
            font-size: 0.78rem;
            font-weight: 700;
            margin: -0.38rem 0 0.66rem 0;
        }}

        .metric-card, .panel {{
            background: rgba(255,255,255,0.92);
            border: 1px solid {colors["border"]};
            border-radius: 18px;
            padding: 1.05rem 1.12rem;
            min-height: 118px;
            box-shadow: 0 18px 45px {colors["shadow"]};
        }}

        .metric-card {{
            position: relative;
            overflow: hidden;
            border-top: 0;
            transition: transform 160ms ease, box-shadow 160ms ease;
        }}

        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 24px 58px rgba(15,46,110,0.14);
        }}

        .metric-card:before {{
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(135deg, rgba(37,99,235,0.10), transparent 42%),
                        linear-gradient(315deg, rgba(5,185,131,0.10), transparent 38%);
            pointer-events: none;
        }}

        .metric-card:after {{
            content: "";
            position: absolute;
            right: 1rem;
            top: 1rem;
            width: 42px;
            height: 42px;
            border-radius: 14px;
            background: linear-gradient(135deg, {colors["accent"]}, {colors["mint"]});
            opacity: 0.92;
            box-shadow: 0 16px 30px rgba(37,99,235,0.20);
        }}

        .metric-label {{
            color: {colors["muted"]};
            font-size: 0.76rem;
            font-weight: 900;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
            position: relative;
            z-index: 1;
        }}

        .metric-value {{
            color: {colors["text"]};
            font-size: 1.78rem;
            line-height: 1.1;
            font-weight: 900;
            letter-spacing: -0.01em;
            max-width: calc(100% - 50px);
            position: relative;
            z-index: 1;
        }}

        .metric-sub {{
            color: {colors["muted"]};
            font-size: 0.86rem;
            margin-top: 0.45rem;
            position: relative;
            z-index: 1;
        }}

        .risk-pill {{
            display: inline-flex;
            align-items: center;
            padding: 0.42rem 0.72rem;
            border-radius: 999px;
            color: white;
            font-size: 0.86rem;
            font-weight: 900;
            box-shadow: 0 10px 24px rgba(15,46,110,0.15);
        }}

        .section-title {{
            color: {colors["text"]};
            font-size: 1.08rem;
            font-weight: 900;
            margin: 0.45rem 0 0.85rem 0;
            letter-spacing: -0.01em;
        }}

        [data-testid="stPlotlyChart"] {{
            background: rgba(255,255,255,0.90);
            border: 1px solid {colors["border"]};
            border-radius: 20px;
            padding: 0.4rem;
            box-shadow: 0 18px 45px {colors["shadow"]};
        }}

        [data-testid="stImage"] {{
            background: rgba(255,255,255,0.9);
            border: 1px solid {colors["border"]};
            border-radius: 20px;
            padding: 0.75rem;
            box-shadow: 0 18px 45px {colors["shadow"]};
        }}

        div[data-testid="stDataFrame"] {{
            border: 1px solid {colors["border"]};
            border-radius: 18px;
            overflow: hidden;
            background: {colors["table_bg"]};
            color: {colors["table_text"]};
            box-shadow: 0 18px 45px {colors["shadow"]};
        }}

        div[data-testid="stDataFrame"] * {{
            font-family: "Raleway", Arial, sans-serif !important;
        }}

        label, .stSelectbox label, .stNumberInput label, .stMultiSelect label {{
            color: {colors["text"]} !important;
            font-size: 0.9rem !important;
            font-weight: 800 !important;
            font-family: "Raleway", Arial, sans-serif !important;
        }}

        input, textarea, select, [data-baseweb="select"] > div, [data-baseweb="input"] > div {{
            background: {colors["surface"]} !important;
            border-color: {colors["border"]} !important;
            color: {colors["text"]} !important;
            border-radius: 12px !important;
            font-family: "Raleway", Arial, sans-serif !important;
            box-shadow: 0 10px 26px rgba(15,46,110,0.04) !important;
        }}

        input:focus, textarea:focus, [data-baseweb="select"] > div:focus-within,
        [data-baseweb="input"] > div:focus-within {{
            border-color: {colors["accent"]} !important;
            box-shadow: 0 0 0 4px rgba(37,99,235,0.11) !important;
        }}

        input::placeholder {{
            color: {colors["muted"]} !important;
            opacity: 0.78 !important;
        }}

        [data-testid="InputInstructions"] {{
            display: none !important;
            visibility: hidden !important;
        }}

        button, button * {{
            font-family: "Raleway", Arial, sans-serif !important;
        }}

        .stButton > button, .stFormSubmitButton > button {{
            background: linear-gradient(135deg, {colors["accent"]}, {colors["accent_hover"]});
            color: white;
            border: 1px solid {colors["accent"]};
            border-radius: 999px;
            font-weight: 900;
            padding: 0.65rem 1.15rem;
            box-shadow: 0 15px 28px rgba(37,99,235,0.24);
        }}

        .stButton > button:hover, .stFormSubmitButton > button:hover {{
            background: linear-gradient(135deg, {colors["accent_hover"]}, #153eaa);
            border-color: {colors["accent_hover"]};
            color: white;
            transform: translateY(-1px);
        }}

        [data-testid="stNumberInput"] button {{
            background: {colors["accent"]} !important;
            border-color: {colors["accent"]} !important;
            color: white !important;
            border-radius: 0 !important;
        }}

        [data-testid="stNumberInput"] button:hover {{
            background: {colors["accent_hover"]} !important;
            border-color: {colors["accent_hover"]} !important;
            color: white !important;
        }}

        [data-baseweb="tag"] {{
            background: {colors["accent"]} !important;
            color: white !important;
            border-radius: 999px !important;
            font-family: "Raleway", Arial, sans-serif !important;
        }}

        hr {{
            border-color: {colors["border"]};
        }}

        .light-table-shell {{
            border: 1px solid {colors["border"]};
            background: {colors["table_bg"]};
            width: 100%;
            overflow: hidden;
            line-height: 0;
            margin: 0;
            display: flex;
            border-radius: 18px;
            box-shadow: 0 18px 45px {colors["shadow"]};
        }}

        .light-table {{
            width: 100%;
            min-width: 100%;
            table-layout: fixed;
            border-collapse: collapse;
            color: {colors["table_text"]};
            background: {colors["table_bg"]};
            font-family: "Raleway", Arial, sans-serif;
            font-size: 0.9rem;
            line-height: 1.3;
            margin: 0;
            flex: 1 1 auto;
        }}

        .light-table th {{
            background: {colors["table_header"]};
            color: {colors["table_muted"]};
            border-bottom: 1px solid {colors["border"]};
            border-right: 0;
            padding: 0.72rem 0.62rem;
            text-align: left;
            white-space: normal;
            font-weight: 900;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }}

        .light-table td {{
            border-bottom: 1px solid {colors["border"]};
            border-right: 0;
            padding: 0.64rem 0.62rem;
            white-space: normal;
            overflow-wrap: break-word;
            vertical-align: top;
        }}

        .light-table tr:nth-child(even) td {{
            background: {colors["table_alt"]};
        }}

        .light-table tbody tr:hover td {{
            background: {colors["accent_soft"]};
        }}

        .light-table td.col-risk-tier {{
            color: {colors["accent"]};
            font-weight: 900;
        }}

        .light-table tbody tr:last-child td {{
            border-bottom: 0;
        }}

        .light-table .col-customer-id,
        .light-table .col-expected-impact {{
            white-space: nowrap;
        }}

        .light-table .col-geography {{
            width: 90px;
            max-width: 89px;
            white-space: nowrap;
        }}

        .light-table .col-risk-tier {{
            width: 86px;
            max-width: 86px;
            white-space: nowrap;
        }}

        .light-table .col-age {{
            width: 48px;
            max-width: 48px;
            white-space: nowrap;
        }}

        .light-table td.col-churn-probability,
        .light-table td.col-priority-score,
        .light-table td.col-revenue-at-risk,
        .light-table td.col-recoverable-revenue,
        .light-table td.col-value {{
            text-align: right;
            white-space: nowrap;
        }}

        .light-table th.col-revenue-at-risk,
        .light-table th.col-recoverable-revenue,
        .light-table th.col-recommended-action,
        .light-table th.col-recommendation-reason {{
            overflow-wrap: normal;
            word-break: normal;
        }}

        .light-table td.col-recommended-action,
        .light-table td.col-recommendation-reason {{
            padding-top: 0.75rem;
            padding-bottom: 0.75rem;
            line-height: 1.4;
        }}

        @media (max-width: 900px) {{
            .block-container {{
                padding-left: 1rem;
                padding-right: 1rem;
            }}

            .hero {{
                padding: 1.15rem;
                border-radius: 18px;
            }}

            .hero:before, .hero:after {{
                display: none;
            }}

            .metric-card, .panel {{
                border-radius: 16px;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_customers() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "final_customers.csv")


@st.cache_data
def load_model_results() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "model_results.csv")


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "feature_importance.csv")


@st.cache_data
def load_shap_importance() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "shap_feature_importance.csv")


@st.cache_resource
def load_artifacts() -> dict:
    return {
        "model": joblib.load(MODEL_DIR / "xgboost_final.joblib"),
        "models": {
            "Logistic Regression": joblib.load(MODEL_DIR / "logistic_regression.joblib"),
            "Random Forest": joblib.load(MODEL_DIR / "random_forest.joblib"),
            "XGBoost": joblib.load(MODEL_DIR / "xgboost_final.joblib"),
        },
        "scaler": joblib.load(MODEL_DIR / "scaler.joblib"),
        "feature_names": joblib.load(MODEL_DIR / "feature_names.joblib"),
        "encoders": {
            "gender": joblib.load(MODEL_DIR / "gender_encoder.joblib"),
            "geography": joblib.load(MODEL_DIR / "geography_encoder.joblib"),
            "wealth_tier": joblib.load(MODEL_DIR / "wealth_tier_encoder.joblib"),
            "age_group": joblib.load(MODEL_DIR / "age_group_encoder.joblib"),
        },
    }


def money(value: float) -> str:
    return f"${value:,.0f}"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def assign_risk_tier(probability: float) -> str:
    if probability < 0.30:
        return "Low Risk"
    if probability < 0.60:
        return "Medium Risk"
    if probability < 0.80:
        return "High Risk"
    return "Critical"


def age_group(age: float) -> str:
    if age < 30:
        return "Young"
    if age < 45:
        return "Mid-Career"
    if age < 60:
        return "Senior"
    return "Retired"


def wealth_tier(balance: float, salary: float) -> str:
    value = (balance * 0.03) + (salary * 0.01)
    if value >= 7000:
        return "Premium"
    if value >= 2500:
        return "Mid"
    return "Low"


def encode_value(encoder, value: str) -> int:
    classes = [str(item) for item in encoder.classes_]
    if value in classes:
        return int(encoder.transform([value])[0])
    return int(encoder.transform([classes[0]])[0])


def build_features(inputs: dict) -> pd.DataFrame:
    salary = max(float(inputs["estimated_salary"]), 1.0)
    tenure = max(float(inputs["tenure_years"]), 0.0)
    balance = float(inputs["account_balance"])
    products = float(inputs["num_products"])
    transactions = float(inputs["num_transactions_6m"])
    avg_amount = float(inputs["avg_transaction_amount"])
    complaints = float(inputs["num_complaints"])
    logins = float(inputs["mobile_app_logins_monthly"])
    net_banking = float(inputs["net_banking_active"])
    active = float(inputs["is_active_member"])
    credit_card = float(inputs["has_credit_card"])
    emi_ratio = float(inputs["emi_to_income_ratio"])
    loan_default = float(inputs["loan_default_history"])
    has_loan = float(inputs["has_loan"])
    credit_score = float(inputs["credit_score"])
    age = float(inputs["age"])

    balance_salary_ratio = balance / salary
    products_per_tenure = products / (tenure + 1)
    digital_engagement_score = min(100, (logins * 1.6) + (net_banking * 18) + min(transactions, 80) * 0.7)
    customer_value = min(1, ((balance * 0.0000018) + (salary * 0.000001) + (products * 0.06) + (transactions * 0.0015)))
    revenue_at_risk = max(0, (balance * 0.035) + (avg_amount * 6) + (products * 900))
    age_tenure_ratio = age / (tenure + 1)
    zero_balance_flag = int(balance <= 100)
    single_product_flag = int(products <= 1)
    credit_engagement_mismatch = int((credit_card == 1) and (active == 0))
    complaint_risk_score = complaints * max(float(inputs["last_interaction_days_ago"]), 1)
    low_digital_activity = int(digital_engagement_score < 40)
    financial_stress_flag = int((emi_ratio > 0.45) or (loan_default == 1))
    derived_wealth_tier = wealth_tier(balance, salary)
    derived_age_group = age_group(age)
    churn_risk_score = (
        single_product_flag * 15
        + (1 - active) * 25
        + low_digital_activity * 20
        + financial_stress_flag * 20
        + min(complaints * 8, 25)
    )
    high_value_customer = int(customer_value > 0.6)
    high_churn_risk = int(churn_risk_score >= 50)
    priority_score = customer_value * (revenue_at_risk / 250000) * max(churn_risk_score / 100, 0.05)

    raw = {
        "age": age,
        "gender": inputs["gender"],
        "geography": inputs["geography"],
        "tenure_years": tenure,
        "num_products": products,
        "has_credit_card": credit_card,
        "is_active_member": active,
        "credit_score": credit_score,
        "account_balance": balance,
        "estimated_salary": salary,
        "num_transactions_6m": transactions,
        "avg_transaction_amount": avg_amount,
        "num_complaints": complaints,
        "last_interaction_days_ago": float(inputs["last_interaction_days_ago"]),
        "mobile_app_logins_monthly": logins,
        "net_banking_active": net_banking,
        "branch_visits_year": float(inputs["branch_visits_year"]),
        "has_loan": has_loan,
        "loan_default_history": loan_default,
        "emi_to_income_ratio": emi_ratio,
        "balance_salary_ratio": balance_salary_ratio,
        "products_per_tenure": products_per_tenure,
        "digital_engagement_score": digital_engagement_score,
        "customer_value": customer_value,
        "revenue_at_risk": revenue_at_risk,
        "age_tenure_ratio": age_tenure_ratio,
        "zero_balance_flag": zero_balance_flag,
        "single_product_flag": single_product_flag,
        "credit_engagement_mismatch": credit_engagement_mismatch,
        "complaint_risk_score": complaint_risk_score,
        "low_digital_activity": low_digital_activity,
        "financial_stress_flag": financial_stress_flag,
        "wealth_tier": derived_wealth_tier,
        "age_group": derived_age_group,
        "churn_risk_score": churn_risk_score,
        "priority_score": priority_score,
        "high_value_customer": high_value_customer,
        "high_churn_risk": high_churn_risk,
    }
    return pd.DataFrame([raw])


def prepare_for_model(feature_row: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    encoded = feature_row.copy()
    for col, encoder in artifacts["encoders"].items():
        encoded[col] = encoded[col].astype(str).map(lambda value: encode_value(encoder, value))
    return encoded[artifacts["feature_names"]]


def recommendation(row: pd.Series, probability: float) -> tuple[str, str, str]:
    risk = assign_risk_tier(probability)
    if row["customer_value"] > 0.6 and probability > 0.5:
        return "Assign Dedicated Relationship Manager", "High-value customer at material churn risk", "High"
    if row["num_complaints"] > 2:
        return "Priority Complaint Resolution + Follow-up", "Repeated complaints strongly correlate with churn", "High"
    if row["financial_stress_flag"] == 1:
        return "Flexible EMI / Loan Restructuring Offer", "Customer shows financial stress indicators", "Medium"
    if row["is_active_member"] == 0:
        return "Loyalty Rewards / Cashback Activation Offer", "Inactive customer needs a clear reactivation trigger", "High"
    if row["digital_engagement_score"] < 40:
        return "Digital Re-engagement Campaign", "Low digital activity indicates disengagement", "Medium"
    if row["single_product_flag"] == 1:
        return "Cross-Sell Credit Card or Investment Product", "Single-product customers have weaker retention stickiness", "Medium"
    if risk in {"High Risk", "Critical"}:
        return "Standard Retention Email Campaign", "High model risk requires proactive retention contact", "Medium"
    return "Periodic Relationship Nurture", "Customer is currently stable but should stay engaged", "Low"


def parse_number(value: str | None, label: str, errors: list[str], *, integer: bool = False, minimum: float | None = None) -> float | int | None:
    if value is None or not str(value).strip():
        errors.append(f"{label} is required.")
        return None
    try:
        number = float(str(value).replace(",", "").strip())
    except ValueError:
        errors.append(f"{label} must be a number.")
        return None
    if minimum is not None and number < minimum:
        errors.append(f"{label} must be at least {minimum:g}.")
        return None
    if integer:
        if not number.is_integer():
            errors.append(f"{label} must be a whole number.")
            return None
        return int(number)
    return number


def field_label(label: str, required: bool = True) -> None:
    star = '<span class="required-star">*</span>' if required else ""
    st.markdown(f'<div class="required-label">{escape(label)}{star}</div>', unsafe_allow_html=True)


def field_error(field_errors: dict[str, str], key: str) -> None:
    if key in field_errors:
        st.markdown(f'<div class="field-error">{escape(field_errors[key])}</div>', unsafe_allow_html=True)


def required_text_input(label: str, key: str, placeholder: str, field_errors: dict[str, str]) -> str:
    field_label(label)
    value = st.text_input(label, key=key, placeholder=placeholder, label_visibility="collapsed")
    field_error(field_errors, key)
    return value


def required_selectbox(label: str, key: str, options: list, placeholder: str, field_errors: dict[str, str]):
    field_label(label)
    value = st.selectbox(label, options, index=None, key=key, placeholder=placeholder, label_visibility="collapsed")
    field_error(field_errors, key)
    return value


def parse_required_float(
    value: str | None,
    label: str,
    key: str,
    field_errors: dict[str, str],
    *,
    minimum: float | None = None,
) -> float | None:
    if value is None or not str(value).strip():
        field_errors[key] = "Required"
        return None
    try:
        number = float(str(value).replace(",", "").strip())
    except ValueError:
        field_errors[key] = "Enter a valid number"
        return None
    if minimum is not None and number < minimum:
        field_errors[key] = f"Must be at least {minimum:g}"
        return None
    return number


def parse_input_float(value: str | None, default: float = 0.0, *, minimum: float | None = None) -> tuple[float, str | None]:
    if value is None or not str(value).strip():
        return default, None
    try:
        number = float(str(value).replace(",", "").strip())
    except ValueError:
        return default, "Enter a valid number"
    if minimum is not None and number < minimum:
        return default, f"Must be at least {minimum:g}"
    return number, None


def add_enter_to_next_field_script() -> None:
    components.html(
        """
        <script>
        (() => {
          const doc = window.parent.document;
          if (doc.body.dataset.retainxEnterNextReady === "true") return;
          doc.body.dataset.retainxEnterNextReady = "true";

          const isVisible = (field) => {
            const rect = field.getBoundingClientRect();
            return rect.width > 0 && rect.height > 0 && field.offsetParent !== null;
          };

          const formFields = () => {
            const fields = Array.from(doc.querySelectorAll(
              'input:not([type="hidden"]):not([disabled]), textarea:not([disabled])'
            )).filter(isVisible);

            return fields.sort((a, b) => {
              const ar = a.getBoundingClientRect();
              const br = b.getBoundingClientRect();
              const aLeftColumn = ar.left < window.parent.innerWidth / 2 ? 0 : 1;
              const bLeftColumn = br.left < window.parent.innerWidth / 2 ? 0 : 1;
              if (aLeftColumn !== bLeftColumn) return aLeftColumn - bLeftColumn;
              if (Math.abs(ar.top - br.top) > 8) return ar.top - br.top;
              return ar.left - br.left;
            });
          };

          doc.addEventListener("keydown", (event) => {
            if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
            const active = doc.activeElement;
            if (!active || active.tagName.toLowerCase() === "textarea") return;
            if (!active.matches('input:not([type="hidden"]):not([disabled])')) return;

            const fields = formFields();
            const currentIndex = fields.indexOf(active);
            if (currentIndex === -1) return;

            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();

            const next = fields[currentIndex + 1];
            if (next) {
              next.focus();
              if (typeof next.select === "function") next.select();
            }
          }, true);
        })();
        </script>
        """,
        height=0,
    )


def gender_for_model(gender: str) -> str:
    return {"Male": "M", "Female": "F", "Other": "Unknown"}[gender]


def page_header(label: str, title: str, subtitle: str) -> None:
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    label_html = f'<div class="eyebrow">{label}</div>' if label else ""
    st.markdown(
        f"""
        <div class="hero">
          {label_html}
          <h1>{title}</h1>
          {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


COLUMN_LABELS = {
    "customer_id": "Customer ID",
    "geography": "Region",
    "age": "Age",
    "churn_probability": "Churn Probability",
    "risk_tier": "Risk Tier",
    "priority_score": "Priority Score",
    "revenue_at_risk": "Revenue At Risk",
    "recoverable_revenue": "Recoverable Revenue",
    "recommended_action": "Recommended Action",
    "recommendation_reason": "Recommendation Reason",
    "expected_impact": "Expected<br>Impact",
}


def column_css_class(column: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in str(column)).strip("-")
    return f"col-{cleaned}"


def light_cell_html(column: str, value) -> str:
    progress_columns = {"churn_probability", "Churn Probability"}
    money_columns = {"revenue_at_risk", "recoverable_revenue", "Revenue At Risk", "Recoverable Revenue"}

    if column in progress_columns and pd.notna(value):
        return escape(f"{float(value):.1%}")
    if column in money_columns and pd.notna(value):
        return escape(money(float(value)))
    if column in {"risk_tier", "Risk Tier"} and value == "Medium Risk":
        return "Medium<br>Risk"
    if column == "priority_score" and pd.notna(value):
        return escape(f"{float(value):.4f}")
    if isinstance(value, float):
        return escape(f"{value:.4f}".rstrip("0").rstrip("."))
    return escape("" if pd.isna(value) else str(value))


def light_dataframe(data: pd.DataFrame, hide_index: bool = False) -> None:
    if not hide_index and (data.index.name is not None or not isinstance(data.index, pd.RangeIndex)):
        data = data.reset_index()
    rows = []
    for _, row in data.iterrows():
        cells = "".join(
            f'<td class="{column_css_class(column)}">{light_cell_html(column, row[column])}</td>'
            for column in data.columns
        )
        rows.append(f"<tr>{cells}</tr>")

    headers = "".join(
        f'<th class="{column_css_class(column)}">{COLUMN_LABELS.get(str(column), escape(str(column)))}</th>'
        for column in data.columns
    )
    st.markdown(
        f'<div class="light-table-shell"><table class="light-table">'
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        f"</table></div>",
        unsafe_allow_html=True,
    )


def themed_dataframe(data: pd.DataFrame, **kwargs) -> None:
    light_dataframe(data, hide_index=kwargs.get("hide_index", False))


def model_input_summary(feature_row: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "age": "Age",
        "gender": "Gender",
        "geography": "Region",
        "num_products": "Number of Products",
        "is_active_member": "Active Member",
        "credit_score": "Credit Score",
        "account_balance": "Account Balance",
        "num_complaints": "Complaints",
        "last_interaction_days_ago": "Days Since Last Interaction",
        "mobile_app_logins_monthly": "Monthly Mobile App Logins",
        "net_banking_active": "Net Banking Active",
        "digital_engagement_score": "Digital Engagement Score",
        "revenue_at_risk": "Revenue At Risk",
        "financial_stress_flag": "Financial Stress Flag",
        "churn_risk_score": "Churn Risk Score",
        "priority_score": "Priority Score",
    }
    display = feature_row.iloc[0][list(labels)].rename(index=labels).reset_index()
    display.columns = ["Input / Derived Feature", "Value"]
    yes_no_rows = {
        "Has Credit Card",
        "Active Member",
        "Net Banking Active",
        "Has Loan",
        "Loan Default History",
        "Financial Stress Flag",
    }
    display["Value"] = display.apply(
        lambda row: "Yes" if row["Input / Derived Feature"] in yes_no_rows and float(row["Value"]) == 1 else (
            "No" if row["Input / Derived Feature"] in yes_no_rows else row["Value"]
        ),
        axis=1,
    )
    return display


def style_plot(fig: go.Figure) -> go.Figure:
    colors = st.session_state.get("theme_colors", theme_colors())
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family='"Raleway", Arial, sans-serif', color=colors["chart_text"]),
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, color=colors["chart_text"])
    fig.update_yaxes(gridcolor=colors["chart_grid"], zeroline=False, color=colors["chart_text"])
    return fig


def executive_overview(df: pd.DataFrame) -> None:
    page_header(
        "Executive Control Room",
        "RetainX Customer Retention Intelligence",
        "A live business view of churn risk, revenue exposure, recoverable value, and where retention teams should focus first.",
    )

    tier_options = ["Critical", "High Risk", "Medium Risk"]
    selected_tiers = st.multiselect(
        "Risk Segment",
        tier_options,
        default=["Critical", "High Risk", "Medium Risk"],
        help="Filter the overview charts and top customer queue by risk tier.",
    )
    chart_df = df[df["risk_tier"].isin(selected_tiers)] if selected_tiers else df

    high_risk = chart_df["risk_tier"].isin(["High Risk", "Critical"]).sum()
    cols = st.columns(4)
    with cols[0]:
        metric_card("Total Customers", f"{len(chart_df):,}", "Customers in selected risk segment")
    with cols[1]:
        metric_card("High Risk Customers", f"{high_risk:,}", "High Risk + Critical customers")
    with cols[2]:
        metric_card("Revenue At Risk", money(chart_df["revenue_at_risk"].sum()), "Estimated exposed relationship value")
    with cols[3]:
        metric_card("Recoverable Revenue", money(chart_df["recoverable_revenue"].sum()), "Modeled recoverable opportunity")

    st.divider()

    left, right = st.columns([1.05, 0.95])
    risk_counts = chart_df["risk_tier"].value_counts().reindex(RISK_COLORS.keys()).fillna(0).reset_index()
    risk_counts.columns = ["risk_tier", "customers"]

    with left:
        st.markdown('<div class="section-title">Customer Risk Distribution</div>', unsafe_allow_html=True)
        fig = px.pie(
            risk_counts,
            values="customers",
            names="risk_tier",
            hole=0.48,
            color="risk_tier",
            color_discrete_map=RISK_COLORS,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=430, margin=dict(l=16, r=16, t=38, b=18))
        st.plotly_chart(style_plot(fig), use_container_width=True)

    with right:
        st.markdown('<div class="section-title">Average Churn Probability by Region</div>', unsafe_allow_html=True)
        geo = chart_df.groupby("geography", as_index=False)["churn_probability"].mean().sort_values("churn_probability", ascending=False)
        fig = px.bar(
            geo,
            x="geography",
            y="churn_probability",
            color="churn_probability",
            color_continuous_scale=["#d9e6e1", "#1b8a8f", "#17324d"],
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(height=430)
        st.plotly_chart(style_plot(fig), use_container_width=True)

    st.markdown('<div class="section-title">Top Customers To Save</div>', unsafe_allow_html=True)
    show = chart_df.sort_values("priority_score", ascending=False)[
        [
            "customer_id",
            "churn_probability",
            "risk_tier",
            "revenue_at_risk",
            "recoverable_revenue",
            "recommended_action",
            "expected_impact",
        ]
    ].head(10)
    themed_dataframe(
        show,
        use_container_width=True,
        hide_index=True,
        column_config={
            "churn_probability": st.column_config.ProgressColumn("Churn Probability", min_value=0, max_value=1, format="%.2f"),
            "revenue_at_risk": st.column_config.NumberColumn("Revenue At Risk", format="$%.0f"),
            "recoverable_revenue": st.column_config.NumberColumn("Recoverable Revenue", format="$%.0f"),
        },
    )


def customer_intelligence(df: pd.DataFrame) -> None:
    page_header(
        "Customer Intelligence",
        "Prioritize customers before churn becomes revenue loss",
        "Filter by risk tier, impact, and region to inspect model-backed retention actions customer by customer.",
    )

    filters = st.columns([1, 1, 1])
    with filters[0]:
        tier = st.selectbox("Risk Tier", ["All"] + list(RISK_COLORS.keys()), index=0)
    with filters[1]:
        impact = st.selectbox("Expected Impact", ["All"] + sorted(df["expected_impact"].dropna().unique().tolist()), index=0)
    with filters[2]:
        geography = st.selectbox("Region", ["All"] + sorted(df["geography"].dropna().unique().tolist()), index=0)

    view = df.copy()
    if tier != "All":
        view = view[view["risk_tier"] == tier]
    if impact != "All":
        view = view[view["expected_impact"] == impact]
    if geography != "All":
        view = view[view["geography"] == geography]

    cols = st.columns(4)
    with cols[0]:
        metric_card("Customers In View", f"{len(view):,}", "After active filters")
    with cols[1]:
        metric_card("Avg Churn Probability", pct(view["churn_probability"].mean() if len(view) else 0), "Mean model score")
    with cols[2]:
        metric_card("Revenue At Risk", money(view["revenue_at_risk"].sum()), "Filtered portfolio")
    with cols[3]:
        metric_card("Recoverable Revenue", money(view["recoverable_revenue"].sum()), "Potential save value")

    st.markdown('<div class="section-title">Customer Action Queue</div>', unsafe_allow_html=True)
    table_cols = [
        "customer_id",
        "geography",
        "age",
        "churn_probability",
        "risk_tier",
        "priority_score",
        "revenue_at_risk",
        "recoverable_revenue",
        "recommended_action",
        "recommendation_reason",
        "expected_impact",
    ]
    themed_dataframe(
        view.sort_values(["priority_score", "churn_probability"], ascending=False)[table_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "churn_probability": st.column_config.ProgressColumn("Churn Probability", min_value=0, max_value=1, format="%.2f"),
            "priority_score": st.column_config.NumberColumn("Priority Score", format="%.4f"),
            "revenue_at_risk": st.column_config.NumberColumn("Revenue At Risk", format="$%.0f"),
            "recoverable_revenue": st.column_config.NumberColumn("Recoverable Revenue", format="$%.0f"),
        },
    )


def prediction_engine(df: pd.DataFrame) -> None:
    add_enter_to_next_field_script()
    page_header(
        "AI Prediction Engine",
        "Predict churn risk for a single customer",
        "Enter a customer's profile and RetainX will score churn probability, revenue exposure, and the next best retention action.",
    )

    artifacts = load_artifacts()
    geography_options = sorted(df["geography"].dropna().unique().tolist())
    gender_options = ["Male", "Female", "Other"]
    field_errors = st.session_state.get("prediction_field_errors", {})

    st.markdown('<div class="form-shell">', unsafe_allow_html=True)
    with st.form("prediction_form"):
        st.markdown('<div class="form-title">Customer Information</div>', unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            credit_score = required_text_input("Credit Score", "credit_score", "Enter credit score", field_errors)
            geography = required_selectbox("Region", "geography", geography_options, "Select region", field_errors)
            gender = required_selectbox("Gender", "gender", gender_options, "Select gender", field_errors)
            age = required_text_input("Age", "age", "Enter age", field_errors)
            tenure_years = required_text_input("Years as Bank Customer", "tenure_years", "Enter years", field_errors)
            account_balance = required_text_input("Account Balance", "account_balance", "Enter balance", field_errors)
            num_transactions_6m = required_text_input("Transactions in Last 6 Months", "num_transactions_6m", "Enter transactions", field_errors)
            avg_transaction_amount = required_text_input("Average Transaction Amount", "avg_transaction_amount", "Enter amount", field_errors)
            num_complaints = required_text_input("Complaints", "num_complaints", "Enter complaints", field_errors)
        with right:
            num_products = required_text_input("Number of Products Used", "num_products", "Enter products", field_errors)
            has_credit_card = required_selectbox("Has an Active Credit Card", "has_credit_card", ["Yes", "No"], "Select option", field_errors)
            is_active_member = required_selectbox("Is an Active Bank Member", "is_active_member", ["Yes", "No"], "Select option", field_errors)
            estimated_salary = required_text_input("Estimated Salary", "estimated_salary", "Enter salary", field_errors)
            has_loan = required_selectbox("Has Loan", "has_loan", ["Yes", "No"], "Select option", field_errors)
            loan_default_history = required_selectbox("Loan Default History", "loan_default_history", ["No", "Yes"], "Select option", field_errors)
            last_interaction_days_ago = required_text_input("Days Since Last Interaction", "last_interaction_days_ago", "Enter days", field_errors)
            mobile_app_logins_monthly = required_text_input("Monthly Mobile App Logins", "mobile_app_logins_monthly", "Enter logins", field_errors)
            net_banking_active = required_selectbox("Net Banking Active", "net_banking_active", ["Yes", "No"], "Select option", field_errors)
            branch_visits_year = required_text_input("Branch Visits Per Year", "branch_visits_year", "Enter visits", field_errors)

        submitted = st.form_submit_button("Submit")
    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        form_errors = {}
        parsed_credit_score = parse_required_float(credit_score, "Credit Score", "credit_score", form_errors, minimum=0)
        parsed_age = parse_required_float(age, "Age", "age", form_errors, minimum=0)
        parsed_tenure_years = parse_required_float(tenure_years, "Years as Bank Customer", "tenure_years", form_errors, minimum=0)
        parsed_account_balance = parse_required_float(account_balance, "Account Balance", "account_balance", form_errors, minimum=0)
        parsed_num_transactions_6m = parse_required_float(num_transactions_6m, "Transactions in Last 6 Months", "num_transactions_6m", form_errors, minimum=0)
        parsed_avg_transaction_amount = parse_required_float(avg_transaction_amount, "Average Transaction Amount", "avg_transaction_amount", form_errors, minimum=0)
        parsed_num_complaints = parse_required_float(num_complaints, "Complaints", "num_complaints", form_errors, minimum=0)
        parsed_num_products = parse_required_float(num_products, "Number of Products Used", "num_products", form_errors, minimum=0)
        parsed_estimated_salary = parse_required_float(estimated_salary, "Estimated Salary", "estimated_salary", form_errors, minimum=0)
        parsed_last_interaction_days_ago = parse_required_float(last_interaction_days_ago, "Days Since Last Interaction", "last_interaction_days_ago", form_errors, minimum=0)
        parsed_mobile_app_logins_monthly = parse_required_float(mobile_app_logins_monthly, "Monthly Mobile App Logins", "mobile_app_logins_monthly", form_errors, minimum=0)
        parsed_branch_visits_year = parse_required_float(branch_visits_year, "Branch Visits Per Year", "branch_visits_year", form_errors, minimum=0)

        required_choices = {
            "geography": geography,
            "gender": gender,
            "has_credit_card": has_credit_card,
            "is_active_member": is_active_member,
            "has_loan": has_loan,
            "loan_default_history": loan_default_history,
            "net_banking_active": net_banking_active,
        }
        for key, value in required_choices.items():
            if value is None:
                form_errors[key] = "Required"
        if parsed_age is not None and parsed_tenure_years is not None and parsed_tenure_years > parsed_age:
            form_errors["tenure_years"] = "Cannot be greater than age"
        if form_errors:
            st.session_state["prediction_field_errors"] = form_errors
            st.rerun()
            return
        st.session_state["prediction_field_errors"] = {}

        inputs = {
            "credit_score": parsed_credit_score,
            "geography": geography,
            "gender": gender_for_model(gender),
            "age": parsed_age,
            "tenure_years": parsed_tenure_years,
            "account_balance": parsed_account_balance,
            "num_products": parsed_num_products,
            "has_credit_card": int(has_credit_card == "Yes"),
            "is_active_member": int(is_active_member == "Yes"),
            "estimated_salary": parsed_estimated_salary,
            "has_loan": int(has_loan == "Yes"),
            "loan_default_history": int(loan_default_history == "Yes"),
            "num_transactions_6m": parsed_num_transactions_6m,
            "avg_transaction_amount": parsed_avg_transaction_amount,
            "num_complaints": parsed_num_complaints,
            "last_interaction_days_ago": parsed_last_interaction_days_ago,
            "mobile_app_logins_monthly": parsed_mobile_app_logins_monthly,
            "net_banking_active": int(net_banking_active == "Yes"),
            "branch_visits_year": parsed_branch_visits_year,
            "emi_to_income_ratio": 0.0,
        }

        feature_row = build_features(inputs)
        model_input = prepare_for_model(feature_row, artifacts)
        scaled = artifacts["scaler"].transform(model_input)
        model_rows = []
        for model_name, model in artifacts["models"].items():
            model_probability = float(model.predict_proba(scaled)[0, 1])
            model_rows.append(
                {
                    "Model Name": model_name,
                    "Prediction": "Likely to Exit" if model_probability >= 0.5 else "Likely to Stay",
                    "Churn Probability": model_probability,
                }
            )
        prediction_table = pd.DataFrame(model_rows)
        probability = float(
            prediction_table.loc[prediction_table["Model Name"] == "XGBoost", "Churn Probability"].iloc[0]
        )
        predicted = int(probability >= 0.5)
        tier = assign_risk_tier(probability)
        action, reason, impact = recommendation(feature_row.iloc[0], probability)
        recoverable = feature_row.loc[0, "revenue_at_risk"] * probability * 0.35

        st.divider()
        st.markdown('<div class="section-title">Model Prediction that Customer is Likely to Exit</div>', unsafe_allow_html=True)
        themed_dataframe(
            prediction_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Churn Probability": st.column_config.ProgressColumn(
                    "Churn Probability",
                    min_value=0,
                    max_value=1,
                    format="%.2f",
                ),
            },
        )

        color = RISK_COLORS[tier]
        st.markdown(
            f"""
            <span class="risk-pill" style="background:{color};">{tier}</span>
            """,
            unsafe_allow_html=True,
        )
        result_cols = st.columns(4)
        with result_cols[0]:
            metric_card("Prediction", "Likely to Exit" if predicted else "Likely to Stay", "XGBoost production model")
        with result_cols[1]:
            metric_card("Churn Probability", pct(probability), "Model probability score")
        with result_cols[2]:
            metric_card("Revenue At Risk", money(feature_row.loc[0, "revenue_at_risk"]), "Derived from balance, products, usage")
        with result_cols[3]:
            metric_card("Recoverable Revenue", money(recoverable), "Using 35% recovery assumption")

        left, right = st.columns([0.85, 1.15])
        with left:
            st.markdown('<div class="section-title">Recommended Retention Move</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="panel">
                  <div class="metric-label">Action</div>
                  <div style="font-size:1.35rem;font-weight:800;">{action}</div>
                  <p style="margin-top:0.7rem;">{reason}</p>
                  <div class="metric-sub">Expected impact: <b>{impact}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            st.markdown('<div class="section-title">Engineered Model Inputs</div>', unsafe_allow_html=True)
            themed_dataframe(model_input_summary(feature_row), use_container_width=True, hide_index=True)


def explainability(df: pd.DataFrame) -> None:
    page_header(
        "Model Explainability",
        "Why the AI flags churn risk",
        "SHAP and model feature importance outputs from the training notebook, shown alongside production model performance.",
    )

    results = load_model_results()
    feat = load_feature_importance().head(12)
    shap = load_shap_importance().head(12)

    left, right = st.columns([0.95, 1.05])
    with left:
        st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)
        themed_dataframe(results, use_container_width=True, hide_index=True)
    with right:
        st.markdown('<div class="section-title">XGBoost Feature Importance</div>', unsafe_allow_html=True)
        fig = px.bar(feat.sort_values("Importance"), x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale=["#d9e6e1", "#1b8a8f", "#17324d"])
        st.plotly_chart(style_plot(fig), use_container_width=True)

    st.markdown('<div class="section-title">SHAP Global Importance</div>', unsafe_allow_html=True)
    fig = px.bar(shap.sort_values("Mean_SHAP_Importance"), x="Mean_SHAP_Importance", y="Feature", orientation="h", color="Mean_SHAP_Importance", color_continuous_scale=["#f4dfc4", "#b57a3a", "#47301b"])
    st.plotly_chart(style_plot(fig), use_container_width=True)

    images = [
        ("SHAP Feature Importance", SHAP_DIR / "shap_feature_importance.png"),
        ("SHAP Beeswarm Analysis", SHAP_DIR / "shap_beeswarm.png"),
        ("Individual Customer Waterfall", SHAP_DIR / "shap_individual_waterfall.png"),
        ("SHAP Dependence Plot", SHAP_DIR / "shap_dependence_plot.png"),
    ]
    cols = st.columns(2)
    for idx, (caption, path) in enumerate(images):
        with cols[idx % 2]:
            if path.exists():
                st.image(str(path), caption=caption, use_container_width=True)


def simulator(df: pd.DataFrame) -> None:
    add_enter_to_next_field_script()
    page_header(
        "What-If Strategy Simulator",
        "Estimate retention campaign impact before spending",
        "Adjust campaign strength and budget to estimate projected churn reduction, recoverable revenue, and ROI.",
    )

    current_churn = float(df["churn_probability"].mean())
    revenue_at_risk = float(df["revenue_at_risk"].sum())
    sim_errors = {}
    field_label("Retention Campaign Strength (%)")
    campaign_strength = st.text_input(
        "Retention Campaign Strength (%)",
        key="campaign_strength",
        placeholder="Enter campaign strength",
        label_visibility="collapsed",
    )
    campaign_strength_value, strength_error = parse_input_float(campaign_strength, minimum=0)
    if strength_error:
        sim_errors["campaign_strength"] = strength_error
        field_error(sim_errors, "campaign_strength")
    campaign_strength_value = min(campaign_strength_value, 100)
    field_label("Retention Budget ($)")
    budget = st.text_input(
        "Retention Budget ($)",
        key="retention_budget",
        placeholder="Enter budget",
        label_visibility="collapsed",
    )
    budget_value, budget_error = parse_input_float(budget, minimum=0)
    if budget_error:
        sim_errors["retention_budget"] = budget_error
        field_error(sim_errors, "retention_budget")
    field_label("Target Risk Tiers")
    target_tiers = st.multiselect(
        "Target Risk Tiers",
        list(RISK_COLORS.keys()),
        label_visibility="collapsed",
    )

    target_df = df[df["risk_tier"].isin(target_tiers)] if target_tiers else df.iloc[0:0]
    effectiveness = min(0.55, campaign_strength_value / 100 * 0.32 + min(budget_value / max(revenue_at_risk, 1), 0.18))
    projected_churn = max(0, current_churn - (target_df["churn_probability"].mean() if len(target_df) else 0) * effectiveness * 0.35)
    estimated_revenue_saved = float((target_df["revenue_at_risk"] * target_df["churn_probability"] * effectiveness).sum())
    roi = ((estimated_revenue_saved - budget_value) / budget_value) * 100 if budget_value else 0

    cols = st.columns(4)
    with cols[0]:
        metric_card("Current Avg Churn", pct(current_churn), "Across full customer base")
    with cols[1]:
        metric_card("Projected Avg Churn", pct(projected_churn), "After simulated campaign")
    with cols[2]:
        metric_card("Estimated ROI", f"{roi:,.1f}%", "Revenue saved vs budget")
    with cols[3]:
        metric_card("Estimated Revenue Saved", money(estimated_revenue_saved), f"{len(target_df):,} targeted customers")

    st.divider()
    curve = pd.DataFrame(
        {
            "Campaign Strength": np.arange(0, 101, 5),
        }
    )
    curve["Revenue Saved"] = curve["Campaign Strength"].map(
        lambda strength: (target_df["revenue_at_risk"] * target_df["churn_probability"] * min(0.55, strength / 100 * 0.32 + min(budget_value / max(revenue_at_risk, 1), 0.18))).sum()
    )
    curve["Projected Churn"] = curve["Campaign Strength"].map(
        lambda strength: max(0, current_churn - (target_df["churn_probability"].mean() if len(target_df) else 0) * min(0.55, strength / 100 * 0.32 + min(budget_value / max(revenue_at_risk, 1), 0.18)) * 0.35)
    )

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Revenue Saved Curve</div>', unsafe_allow_html=True)
        fig = px.line(curve, x="Campaign Strength", y="Revenue Saved", markers=True)
        fig.update_yaxes(tickprefix="$")
        st.plotly_chart(style_plot(fig), use_container_width=True)
    with right:
        st.markdown('<div class="section-title">Projected Churn Curve</div>', unsafe_allow_html=True)
        fig = px.line(curve, x="Campaign Strength", y="Projected Churn", markers=True)
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style_plot(fig), use_container_width=True)


def main() -> None:
    df = load_customers()
    df["risk_tier"] = df["risk_tier"].astype(str)

    with st.sidebar:
        st.markdown('<div class="sidebar-brand">RetainX</div>', unsafe_allow_html=True)
        st.caption("AI-powered retention intelligence")
        page = st.radio(
            "Navigation",
            [
                "AI Prediction Engine",
                "Executive Overview",
                "Customer Intelligence",
                "Model Explainability",
                "What-If Simulator",
            ],
        )

    inject_theme()
    add_enter_to_next_field_script()

    if page == "AI Prediction Engine":
        prediction_engine(df)
    elif page == "Executive Overview":
        executive_overview(df)
    elif page == "Customer Intelligence":
        customer_intelligence(df)
    elif page == "Model Explainability":
        explainability(df)
    else:
        simulator(df)


if __name__ == "__main__":
    main()
