import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------------------------
PAGE_TITLE = "Telecom Customer Churn Prediction"
PAGE_ICON = "💼"

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# ---------------------------------------------------------------------------
# Custom dark theme styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
    }
    body {
        background-color: #080b14;
        color: #e2e8f0;
    }
    .stApp {
        background: #080b14;
    }
    .block-container {
        background-color: #0b1324;
        padding: 2rem 2.2rem 2rem;
        border-radius: 28px;
        box-shadow: 0 28px 80px rgba(0, 0, 0, 0.45);
    }
    .stButton>button {
        background-color: #2563eb;
        color: #ffffff;
        border-radius: 12px;
        height: 3.2rem;
        font-size: 1rem;
        font-weight: 700;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1c64f2;
    }
    .stSelectbox>div>div>div>span,
    .stTextInput>div>input,
    .stNumberInput>div>div>input {
        background-color: #111b33;
        color: #e2e8f0;
        border: 1px solid #1f2a4b;
    }
    .page-header {
        border-radius: 28px;
        padding: 2rem 2rem 1.5rem;
        background: linear-gradient(180deg, rgba(15,23,42,0.95), rgba(10,14,27,0.95));
        border: 1px solid rgba(96,165,250,0.18);
        box-shadow: 0 24px 60px rgba(0,0,0,0.35);
        margin-bottom: 1.75rem;
    }
    .page-header h1 {
        margin-bottom: 0.25rem;
        color: #a5b4fc;
    }
    .page-header p {
        color: #cbd5e1;
        font-size: 1.05rem;
    }
    .section-card {
        border-radius: 24px;
        background-color: #0f172a;
        border: 1px solid rgba(96,165,250,0.12);
        padding: 1.5rem;
        box-shadow: 0 18px 45px rgba(0,0,0,0.20);
        margin-bottom: 1.5rem;
    }
    .dashboard-summary {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
    }
    .dashboard-summary div {
        padding: 1rem 1.2rem;
        border-radius: 18px;
        background: #0b1528;
        border: 1px solid rgba(148,163,184,0.12);
    }
    .result-card {
        border-radius: 24px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        color: #ffffff;
        background-image: linear-gradient(135deg, rgba(20,184,166,0.95), rgba(16,185,129,0.95));
        box-shadow: 0 18px 50px rgba(16,185,129,0.22);
    }
    .result-card.danger {
        background-image: linear-gradient(135deg, rgba(239,68,68,0.95), rgba(220,38,38,0.95));
        box-shadow: 0 18px 50px rgba(239,68,68,0.22);
    }
    .metric-card {
        border-radius: 20px;
        background-color: #0d172f;
        border: 1px solid rgba(96,165,250,0.16);
        padding: 1.3rem 1.2rem;
        min-height: 140px;
    }
    .metric-card h4 {
        margin-bottom: 0.75rem;
        color: #93c5fd;
    }
    .metric-card p {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 0;
    }
    .footer-text {
        color: #94a3b8;
        text-align: center;
        padding: 1rem 0 0.5rem;
    }
    .sidebar .css-1d391kg, .sidebar .css-1wrcr25 {
        background: #090d18 !important;
    }
    .tab-hint {
        display: block;
        margin-bottom: 1rem;
        color: #94a3b8;
    }
    .sidebar-info {
        padding: 1rem 1rem 0.75rem;
        border-radius: 18px;
        background: #09101f;
        border: 1px solid rgba(96,165,250,0.12);
        margin-bottom: 1rem;
    }
    .sidebar-info strong {
        color: #cbd5e1;
    }
    .sidebar-divider {
        margin: 1rem 0;
        border-top: 1px solid rgba(148,163,184,0.12);
    }
    .sidebar-note {
        color: #94a3b8;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar content
# ---------------------------------------------------------------------------
st.sidebar.markdown("<div class='nav-card'><h3>📍 Navigation Menu</h3></div>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "",
    ["Home", "Predict", "About Dataset", "Help & Info"],
    index=1,
)

st.sidebar.markdown("<div class='sidebar-info'>", unsafe_allow_html=True)
st.sidebar.markdown("**Model file:** customer_churn_ann.h5")
st.sidebar.markdown("**Scaler file:** scaler.pkl")
st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
st.sidebar.markdown("**Features:** credit_score, gender, age, tenure, balance, num_of_products, has_credit_card, is_active_member, estimated_salary, geography_Germany, geography_Spain")
st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<span class='sidebar-note'>Use the Predict page for live inference with the same preprocessing used during training.</span>", unsafe_allow_html=True)
st.sidebar.markdown("</div>", unsafe_allow_html=True)
st.sidebar.markdown("Built with ❤️ by Aatequa Ansari")

# ---------------------------------------------------------------------------
# Load model and scaler resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_ann_model(path: str):
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load_model(model_path)


@st.cache_resource(show_spinner=False)
def load_scaler(path: str):
    scaler_path = Path(path)
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    return joblib.load(scaler_path)


try:
    model = load_ann_model("customer_churn_ann.h5")
    scaler = load_scaler("scaler.pkl")
except Exception as error:
    st.error("Unable to load the model or scaler. Please make sure both files exist in the app folder.")
    st.error(f"Details: {error}")
    st.stop()


# ---------------------------------------------------------------------------
# Page rendering helpers
# ---------------------------------------------------------------------------
def render_header():
    st.markdown("<div class='page-header'>", unsafe_allow_html=True)
    st.markdown(f"<h1>{PAGE_TITLE}</h1>", unsafe_allow_html=True)
    st.markdown("<p>Predict whether a customer will churn using a trained neural network and real customer signals.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_footer():
    st.markdown("<div class='footer-text'>Built with ❤️ by Aatequa Ansari.</div>", unsafe_allow_html=True)


def build_feature_vector(
    credit_score: int,
    gender: str,
    age: int,
    tenure: int,
    balance: float,
    num_of_products: int,
    has_credit_card: str,
    is_active_member: str,
    estimated_salary: float,
    geography: str,
) -> pd.DataFrame:
    geography_germany = 1 if geography == "Germany" else 0
    geography_spain = 1 if geography == "Spain" else 0
    gender_male = 1 if gender == "Male" else 0
    has_cr_card = 1 if has_credit_card == "Yes" else 0
    is_active_member_flag = 1 if is_active_member == "Yes" else 0

    age_transformed = np.log1p(age)
    balance_transformed = np.log1p(balance)

    return pd.DataFrame(
        [
            [
                credit_score,
                gender_male,
                age_transformed,
                tenure,
                balance_transformed,
                num_of_products,
                has_cr_card,
                is_active_member_flag,
                estimated_salary,
                geography_germany,
                geography_spain,
            ]
        ],
        columns=[
            "credit_score",
            "gender",
            "age",
            "tenure",
            "balance",
            "num_of_products",
            "has_cr_card",
            "is_active_member",
            "estimated_salary",
            "geography_Germany",
            "geography_Spain",
        ],
    )


def render_home():
    render_header()
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("## Welcome to the Churn Prediction Dashboard", unsafe_allow_html=True)
    st.write(
        "This dashboard evaluates customer churn risk using a pre-trained artificial neural network. Enter customer profile values on the Predict page to see live churn probability, confidence, and retention guidance."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='dashboard-summary'>", unsafe_allow_html=True)
    st.markdown("<div><strong>Model:</strong> Artificial Neural Network</div>", unsafe_allow_html=True)
    st.markdown("<div><strong>Inputs:</strong> 11 numerical and categorical features</div>", unsafe_allow_html=True)
    st.markdown("<div><strong>Preprocessing:</strong> log-transform + standard scaling</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("### Why churn scoring matters", unsafe_allow_html=True)
        st.markdown(
            "<ul>"
            "<li>Identify at-risk customers early.</li>"
            "<li>Build retention campaigns around high-risk profiles.</li>"
            "<li>Use data-driven decisions to reduce churn.</li>"
            "</ul>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("### How the dashboard works", unsafe_allow_html=True)
        st.markdown(
            "<ul>"
            "<li>Input customer information in the Predict page.</li>"
            "<li>The app applies the same transformations used during model training.</li>"
            "<li>Prediction results are presented clearly with confidence and next steps.</li>"
            "</ul>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    render_footer()


def render_prediction_page():
    render_header()
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("## Predict Customer Churn", unsafe_allow_html=True)
    st.markdown("<span class='tab-hint'>Provide customer profile details to generate a churn risk score.</span>", unsafe_allow_html=True)

    with st.form(key="churn_prediction_form"):
        left_col, right_col = st.columns(2)

        with left_col:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650, step=1)
            gender = st.selectbox("Gender", ["Female", "Male"])
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            age = st.number_input("Age", min_value=18, max_value=100, value=38, step=1)
            tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3, step=1)

        with right_col:
            balance = st.number_input("Balance", min_value=0.0, max_value=250000.0, value=60000.0, step=100.0, format="%.2f")
            estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=250000.0, value=52000.0, step=100.0, format="%.2f")
            num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1, step=1)
            has_credit_card = st.selectbox("Has Credit Card", ["Yes", "No"])
            is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])

        submit_col1, submit_col2, submit_col3 = st.columns([1, 2, 1])
        with submit_col2:
            submit_button = st.form_submit_button("Predict Churn Risk")

    if submit_button:
        try:
            feature_vector = build_feature_vector(
                credit_score=credit_score,
                gender=gender,
                age=age,
                tenure=tenure,
                balance=balance,
                num_of_products=num_of_products,
                has_credit_card=has_credit_card,
                is_active_member=is_active_member,
                estimated_salary=estimated_salary,
                geography=geography,
            )

            with st.spinner("Computing churn probability..."):
                scaled_features = scaler.transform(feature_vector)
                prediction = model.predict(scaled_features, verbose=0)

            probability = float(np.asarray(prediction).reshape(-1)[0])
            probability_percent = probability * 100
            will_churn = probability >= 0.5
            confidence_percent = probability_percent if will_churn else 100 - probability_percent
            risk_label = "High" if will_churn else "Low"
            recommendation = (
                "Retain this customer with proactive offers and support."
                if will_churn
                else "Customer appears stable; continue positive engagement."
            )

            if will_churn:
                st.markdown(
                    "<div class='result-card danger'>"
                    f"<h2>❌ Churn Risk: {risk_label}</h2>"
                    f"<p><strong>Probability:</strong> {probability_percent:.2f}%</p>"
                    f"<p><strong>Confidence:</strong> {confidence_percent:.2f}%</p>"
                    f"<p><strong>Recommendation:</strong> {recommendation}</p>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='result-card'>"
                    f"<h2>✅ Churn Risk: {risk_label}</h2>"
                    f"<p><strong>Probability:</strong> {probability_percent:.2f}%</p>"
                    f"<p><strong>Confidence:</strong> {confidence_percent:.2f}%</p>"
                    f"<p><strong>Recommendation:</strong> {recommendation}</p>"
                    "</div>",
                    unsafe_allow_html=True,
                )

            metric_cols = st.columns(4)
            metric_cols[0].markdown(
                "<div class='metric-card'><h4>Risk Level</h4>"
                f"<p>{risk_label}</p></div>",
                unsafe_allow_html=True,
            )
            metric_cols[1].markdown(
                "<div class='metric-card'><h4>Churn Probability</h4>"
                f"<p>{probability_percent:.2f}%</p></div>",
                unsafe_allow_html=True,
            )
            metric_cols[2].markdown(
                "<div class='metric-card'><h4>Confidence</h4>"
                f"<p>{confidence_percent:.2f}%</p></div>",
                unsafe_allow_html=True,
            )
            metric_cols[3].markdown(
                "<div class='metric-card'><h4>Next Step</h4>"
                f"<p>{recommendation}</p></div>",
                unsafe_allow_html=True,
            )

        except Exception as error:
            st.error("Prediction failed. Please check your inputs and try again.")
            st.error(f"Details: {error}")

    render_footer()


def render_about():
    render_header()
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("## About the Dataset", unsafe_allow_html=True)
    st.write(
        "The churn dataset contains customer demographics, account information, and usage signals used to predict whether customers are likely to leave."
    )
    st.markdown(
        "<ul>"
        "<li><strong>Key features:</strong> credit score, age, balance, salary, products, membership status, geography.</li>"
        "<li><strong>Model type:</strong> Artificial neural network trained in Keras.</li>"
        "<li>Preprocessing includes log transformation for skewed values and standard scaling.</li>"
        "</ul>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("### Feature details", unsafe_allow_html=True)
    st.markdown(
        "<ol>"
        "<li><strong>Credit Score:</strong> 300–900</li>"
        "<li><strong>Geography:</strong> France, Germany, Spain</li>"
        "<li><strong>Balance:</strong> Numeric account balance</li>"
        "<li><strong>Estimated Salary:</strong> Annual salary estimate</li>"
        "</ol>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    render_footer()


def render_help():
    render_header()
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("## Help & Instructions", unsafe_allow_html=True)
    st.markdown(
        "<ul>"
        "<li>Select the Predict tab from the sidebar.</li>"
        "<li>Enter customer data into the form fields.</li>"
        "<li>Click Predict Churn Risk to generate the score.</li>"
        "<li>Use the recommendation to plan retention actions.</li>"
        "</ul>",
        unsafe_allow_html=True,
    )
    st.markdown("<p>If the model returns high churn risk, consider targeted campaigns, loyalty perks, or improved customer support.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    render_footer()


if page == "Home":
    render_home()
elif page == "Predict":
    render_prediction_page()
elif page == "About Dataset":
    render_about()
else:
    render_help()
