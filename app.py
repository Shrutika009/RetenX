from pathlib import Path
from html import escape

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


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
        "bg": "#f5f8fb",
        "surface": "#fffdf9",
        "soft": "#eaf1f7",
        "text": "#17120f",
        "muted": "#5f554d",
        "border": "#d8e2ea",
        "sidebar": "#f8fbfd",
        "accent": "#426f98",
        "accent_hover": "#315a80",
        "accent_soft": "#dce8f3",
        "table_header": "#f6f9fc",
        "table_bg": "#fffdf9",
        "table_alt": "#f3f7fb",
        "table_text": "#111111",
        "table_muted": "#6c6259",
        "progress_track": "#dbe7f1",
        "chart_text": "#3f3f3f",
        "chart_grid": "#d7e1ea",
        "shadow": "rgba(41,76,108,0.08)",
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
            background: {colors["bg"]};
            color: {colors["text"]};
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }}

        [data-testid="stHeader"], [data-testid="stToolbar"], header {{
            background: {colors["bg"]} !important;
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
            background: {colors["sidebar"]};
            border-right: 1px solid {colors["border"]};
        }}

        [data-testid="stSidebar"] * {{
            color: {colors["text"]};
        }}

        [data-testid="stSidebar"] [role="radiogroup"] label {{
            padding: 0.5rem 0.65rem;
            border-radius: 4px;
            margin-bottom: 0.15rem;
        }}

        [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
            background: {colors["soft"]};
        }}

        [data-testid="stSidebar"] hr {{
            border-color: {colors["border"]};
        }}

        .sidebar-brand {{
            font-family: "Raleway", Arial, sans-serif;
            font-size: 1.65rem;
            line-height: 1.1;
            font-weight: 800;
            margin: 0 0 0.7rem 0;
        }}

        [role="radio"][aria-checked="true"] div:first-child {{
            background: {colors["accent"]} !important;
            border-color: {colors["accent"]} !important;
        }}

        .hero {{
            padding: 0.7rem 0 1.1rem 0;
        }}

        .eyebrow {{
            color: {colors["accent"]};
            font-size: 1.35rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }}

        .eyebrow:empty {{
            display: none;
        }}

        .hero h1 {{
            color: {colors["text"]};
            font-size: clamp(2rem, 3vw, 3rem);
            line-height: 1.12;
            letter-spacing: 0;
            margin: 0;
            font-weight: 800;
        }}

        .hero p {{
            color: {colors["muted"]};
            max-width: 760px;
            font-size: 1.05rem;
            margin-top: 0.65rem;
        }}

        .form-shell {{
            max-width: 980px;
            margin: 1.2rem auto 1.4rem auto;
        }}

        div[data-testid="stForm"] {{
            background: {colors["surface"]};
            border: 1px solid {colors["border"]};
            border-radius: 10px;
            padding: 1rem 1rem 1.25rem 1rem;
            box-shadow: 0 12px 30px {colors["shadow"]};
        }}

        .form-title {{
            color: {colors["text"]};
            font-size: 1.05rem;
            font-weight: 800;
            margin: 0 0 1.1rem 0;
        }}

        .metric-card, .panel {{
            background: {colors["surface"]};
            border: 1px solid {colors["border"]};
            border-radius: 8px;
            padding: 1rem 1.05rem;
            min-height: 110px;
            box-shadow: 0 10px 28px {colors["shadow"]};
        }}

        .metric-card {{
            border-top: 3px solid {colors["accent"]};
        }}

        .metric-label {{
            color: {colors["muted"]};
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }}

        .metric-value {{
            color: {colors["text"]};
            font-size: 1.75rem;
            line-height: 1.1;
            font-weight: 800;
        }}

        .metric-sub {{
            color: {colors["muted"]};
            font-size: 0.86rem;
            margin-top: 0.45rem;
        }}

        .risk-pill {{
            display: inline-flex;
            align-items: center;
            padding: 0.38rem 0.65rem;
            border-radius: 4px;
            color: white;
            font-size: 0.86rem;
            font-weight: 800;
        }}

        .section-title {{
            color: {colors["text"]};
            font-size: 1.08rem;
            font-weight: 800;
            margin: 0.3rem 0 0.8rem 0;
        }}

        div[data-testid="stDataFrame"] {{
            border: 1px solid {colors["border"]};
            border-radius: 0;
            overflow: hidden;
            background: {colors["table_bg"]};
            color: {colors["table_text"]};
        }}

        div[data-testid="stDataFrame"] * {{
            font-family: "Raleway", Arial, sans-serif !important;
        }}

        label, .stSelectbox label, .stNumberInput label, .stMultiSelect label {{
            color: {colors["text"]} !important;
            font-size: 1rem !important;
            font-weight: 700 !important;
            font-family: "Raleway", Arial, sans-serif !important;
        }}

        input, textarea, select, [data-baseweb="select"] > div, [data-baseweb="input"] > div {{
            background: {colors["surface"]} !important;
            border-color: {colors["border"]} !important;
            color: {colors["text"]} !important;
            border-radius: 7px !important;
            font-family: "Raleway", Arial, sans-serif !important;
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
            background: {colors["accent"]};
            color: white;
            border: 1px solid {colors["accent"]};
            border-radius: 999px;
            font-weight: 800;
            padding: 0.58rem 1rem;
        }}

        .stButton > button:hover, .stFormSubmitButton > button:hover {{
            background: {colors["accent_hover"]};
            border-color: {colors["accent_hover"]};
            color: white;
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
            border-radius: 4px !important;
            font-family: "Raleway", Arial, sans-serif !important;
        }}

        hr {{
            border-color: {colors["border"]};
        }}

        .light-table-shell {{
            border: 1px solid {colors["border"]};
            background: {colors["table_bg"]};
            width: 100%;
            overflow: visible;
            line-height: 0;
            margin: 0;
            display: flex;
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
            border-right: 1px solid {colors["border"]};
            padding: 0.4rem 0.45rem;
            text-align: left;
            white-space: normal;
            font-weight: 700;
        }}

        .light-table td {{
            border-bottom: 1px solid {colors["border"]};
            border-right: 1px solid {colors["border"]};
            padding: 0.34rem 0.45rem;
            white-space: normal;
            overflow-wrap: break-word;
            vertical-align: top;
        }}

        .light-table tr:nth-child(even) td {{
            background: {colors["table_alt"]};
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


def gender_for_model(gender: str) -> str:
    return {"Male": "M", "Female": "F", "Other": "Unknown"}[gender]


def page_header(label: str, title: str, subtitle: str) -> None:
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">{label}</div>
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
    "geography": "Country",
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
        "geography": "Country",
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

    high_risk = df["risk_tier"].isin(["High Risk", "Critical"]).sum()
    cols = st.columns(4)
    with cols[0]:
        metric_card("Total Customers", f"{len(df):,}", "Scored from processed customer data")
    with cols[1]:
        metric_card("High Risk Customers", f"{high_risk:,}", "High Risk + Critical customers")
    with cols[2]:
        metric_card("Revenue At Risk", money(df["revenue_at_risk"].sum()), "Estimated exposed relationship value")
    with cols[3]:
        metric_card("Recoverable Revenue", money(df["recoverable_revenue"].sum()), "Modeled recoverable opportunity")

    st.divider()

    left, right = st.columns([1.05, 0.95])
    risk_counts = df["risk_tier"].value_counts().reindex(RISK_COLORS.keys()).fillna(0).reset_index()
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
        st.plotly_chart(style_plot(fig), use_container_width=True)

    with right:
        st.markdown('<div class="section-title">Average Churn Probability by Geography</div>', unsafe_allow_html=True)
        geo = df.groupby("geography", as_index=False)["churn_probability"].mean().sort_values("churn_probability", ascending=False)
        fig = px.bar(
            geo,
            x="geography",
            y="churn_probability",
            color="churn_probability",
            color_continuous_scale=["#d9e6e1", "#1b8a8f", "#17324d"],
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style_plot(fig), use_container_width=True)

    st.markdown('<div class="section-title">Top Customers To Save</div>', unsafe_allow_html=True)
    show = df.sort_values("priority_score", ascending=False)[
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
        "Filter by risk tier, impact, and geography to inspect model-backed retention actions customer by customer.",
    )

    filters = st.columns([1, 1, 1])
    with filters[0]:
        tier = st.selectbox("Risk Tier", ["All"] + list(RISK_COLORS.keys()), index=0)
    with filters[1]:
        impact = st.selectbox("Expected Impact", ["All"] + sorted(df["expected_impact"].dropna().unique().tolist()), index=0)
    with filters[2]:
        geography = st.selectbox("Geography", ["All"] + sorted(df["geography"].dropna().unique().tolist()), index=0)

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
    page_header(
        "",
        "Enter User Information in the Form Below",
        "",
    )

    artifacts = load_artifacts()
    geography_options = sorted(df["geography"].dropna().unique().tolist())
    gender_options = ["Male", "Female", "Other"]

    st.markdown('<div class="form-shell">', unsafe_allow_html=True)
    with st.form("prediction_form"):
        st.markdown('<div class="form-title">Customer Information</div>', unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            credit_score = st.text_input("Credit Score", placeholder="Enter credit score")
            geography = st.selectbox("Country", geography_options, index=None, placeholder="Select country")
            gender = st.selectbox("Gender", gender_options, index=None, placeholder="Select gender")
            age = st.text_input("Age", placeholder="Enter age")
            tenure_years = st.text_input("Years as Bank Customer", placeholder="Enter years")
            account_balance = st.text_input("Account Balance", placeholder="Enter balance")
            num_transactions_6m = st.text_input("Transactions in Last 6 Months", placeholder="Enter transactions")
            avg_transaction_amount = st.text_input("Average Transaction Amount", placeholder="Enter amount")
            num_complaints = st.text_input("Complaints", placeholder="Enter complaints")
        with right:
            num_products = st.selectbox("Number of Products Used", [1, 2, 3, 4], index=None, placeholder="Select products")
            has_credit_card = st.selectbox("Has an Active Credit Card", ["Yes", "No"], index=None, placeholder="Select option")
            is_active_member = st.selectbox("Is an Active Bank Member", ["Yes", "No"], index=None, placeholder="Select option")
            estimated_salary = st.text_input("Estimated Salary", placeholder="Enter salary")
            has_loan = st.selectbox("Has Loan", ["Yes", "No"], index=None, placeholder="Select option")
            loan_default_history = st.selectbox("Loan Default History", ["No", "Yes"], index=None, placeholder="Select option")
            last_interaction_days_ago = st.text_input("Days Since Last Interaction", placeholder="Enter days")
            mobile_app_logins_monthly = st.text_input("Monthly Mobile App Logins", placeholder="Enter logins")
            net_banking_active = st.selectbox("Net Banking Active", ["Yes", "No"], index=None, placeholder="Select option")
            branch_visits_year = st.text_input("Branch Visits Per Year", placeholder="Enter visits")

        submitted = st.form_submit_button("Submit")
    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        errors = []
        parsed_credit_score = parse_number(credit_score, "Credit Score", errors, integer=True, minimum=0)
        parsed_age = parse_number(age, "Age", errors, integer=True, minimum=0)
        parsed_tenure_years = parse_number(tenure_years, "Years as Bank Customer", errors, integer=True, minimum=0)
        parsed_account_balance = parse_number(account_balance, "Account Balance", errors, minimum=0)
        parsed_num_transactions_6m = parse_number(num_transactions_6m, "Transactions in Last 6 Months", errors, integer=True, minimum=0)
        parsed_avg_transaction_amount = parse_number(avg_transaction_amount, "Average Transaction Amount", errors, minimum=0)
        parsed_num_complaints = parse_number(num_complaints, "Complaints", errors, integer=True, minimum=0)
        parsed_estimated_salary = parse_number(estimated_salary, "Estimated Salary", errors, minimum=0)
        parsed_last_interaction_days_ago = parse_number(last_interaction_days_ago, "Days Since Last Interaction", errors, integer=True, minimum=0)
        parsed_mobile_app_logins_monthly = parse_number(mobile_app_logins_monthly, "Monthly Mobile App Logins", errors, integer=True, minimum=0)
        parsed_branch_visits_year = parse_number(branch_visits_year, "Branch Visits Per Year", errors, integer=True, minimum=0)

        required_choices = {
            "Country": geography,
            "Gender": gender,
            "Number of Products Used": num_products,
            "Has an Active Credit Card": has_credit_card,
            "Is an Active Bank Member": is_active_member,
            "Has Loan": has_loan,
            "Loan Default History": loan_default_history,
            "Net Banking Active": net_banking_active,
        }
        missing = [label for label, value in required_choices.items() if value is None]
        if missing:
            errors.append("Select: " + ", ".join(missing) + ".")
        if parsed_age is not None and parsed_tenure_years is not None and parsed_tenure_years > parsed_age:
            errors.append("Years as Bank Customer cannot be greater than Age.")
        if errors:
            st.warning("Please fix the form before submitting:\n\n" + "\n".join(f"- {error}" for error in errors))
            return

        inputs = {
            "credit_score": parsed_credit_score,
            "geography": geography,
            "gender": gender_for_model(gender),
            "age": parsed_age,
            "tenure_years": parsed_tenure_years,
            "account_balance": parsed_account_balance,
            "num_products": num_products,
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
    page_header(
        "What-If Strategy Simulator",
        "Estimate retention campaign impact before spending",
        "Adjust campaign strength and budget to estimate projected churn reduction, recoverable revenue, and ROI.",
    )

    current_churn = float(df["churn_probability"].mean())
    revenue_at_risk = float(df["revenue_at_risk"].sum())
    campaign_strength = st.number_input("Retention Campaign Strength (%)", min_value=0, max_value=100, value=None, step=1)
    budget = st.number_input("Retention Budget ($)", min_value=0, value=None, step=5000)
    target_tiers = st.multiselect(
        "Target Risk Tiers",
        list(RISK_COLORS.keys()),
    )

    campaign_strength_value = float(campaign_strength or 0)
    budget_value = float(budget or 0)
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
