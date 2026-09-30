import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

st.set_page_config(
    page_title="Sydney Housing Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ---- Custom styling ----
st.markdown("""
    <style>
    .main {
        background-color: #e0e0e0;
    }
    .stApp {
        background-color: #e0e0e0;
    }
    div.stButton > button {
        background-color: #1f4e79;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6em 1.5em;
        border: none;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #163a5f;
        color: white;
    }
    .result-box {
        background-color: #e8f5e9;
        border-left: 6px solid #2e7d32;
        padding: 1.2em;
        border-radius: 8px;
        margin-top: 1.5em;
    }
    .result-price {
        font-size: 2.2em;
        font-weight: 800;
        color: #1b5e20;
        margin: 0;
    }
    h1 {
        color: #1f4e79;
    }

    /* --- Property details form card --- */
    div[data-testid="stForm"] {
        background-color: #eaf2fb;
        border: 1px solid #cfe0f5;
        border-radius: 12px;
        padding: 1.5em 1.5em 0.5em 1.5em;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }

    /* --- Selectbox (Suburb, Property type): white box, black label --- */
    div[data-testid="stSelectbox"] > div > div {
        background-color: #ffffff;
        border: 1px solid #cccccc;
        border-radius: 8px;
    }
    div[data-testid="stSelectbox"] label {
        font-weight: 600;
        color: #000000;
        font-size: 1.05em;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] span {
        color: #000000;
        font-weight: 500;
    }

    /* --- Number inputs (Bedrooms, Bathrooms, Car spaces, Building size, Year built): white box --- */
    div[data-testid="stNumberInput"] input {
        background-color: #ffffff !important;
        border: 1px solid #cccccc !important;
        border-radius: 8px !important;
        color: #000000 !important;
    }
    div[data-testid="stNumberInput"] label {
        font-weight: 600;
        color: #000000;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = joblib.load("sydney_housing_model.pkl")
    meta = joblib.load("model_metadata.pkl")
    return model, meta

model, meta = load_model()

# ---- Header (in a styled box) ----
st.markdown("""
    <div style="
        background-color: #eaf2fb;
        border: 1px solid #cfe0f5;
        border-radius: 12px;
        padding: 1.2em 1.5em;
        margin-bottom: 1.5em;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    ">
        <h1 style="margin:0; color:#1f4e79; white-space: nowrap; font-size: 2.1em;">🏠 Sydney Housing Price Predictor</h1>
        <p style="margin:0.4em 0 0 0; color:#444; font-size:1.05em;">
            Get an instant, data-driven sale price estimate based on real Sydney property sales.
        </p>
    </div>
""", unsafe_allow_html=True)

# ---- Input form (styled as a blue card via CSS above) ----
with st.form("prediction_form"):
    st.markdown("### 📋 Property Details")

    col1, col2 = st.columns(2)
    with col1:
        suburb = st.selectbox("📍 Suburb", meta["suburbs"])
        property_type = st.selectbox("🏘️ Property type", meta["property_types"])
        bedrooms = st.number_input("🛏️ Bedrooms", 0, 10, 3)
        bathrooms = st.number_input("🚿 Bathrooms", 0, 10, 2)
    with col2:
        car_spaces = st.number_input("🚗 Car spaces", 0, 10, 1)
        building_size_sqm = st.number_input("📐 Building size (sqm)", 10, 1500, 120)
        year_built = st.number_input("🏗️ Year built", 1850, datetime.now().year, 2000)

    st.markdown("")
    submitted = st.form_submit_button("🔮 Predict Sale Price")

# ---- Result ----
if submitted:
    property_age = datetime.now().year - year_built
    total_rooms = bedrooms + bathrooms

    input_row = pd.DataFrame([{
        "bedrooms": bedrooms, "bathrooms": bathrooms, "car_spaces": car_spaces,
        "building_size_sqm": building_size_sqm, "property_age": property_age,
        "total_rooms": total_rooms, "suburb": suburb, "property_type": property_type,
    }])

    prediction = model.predict(input_row)[0]

    st.markdown(f"""
        <div class="result-box">
            <p style="margin:0; color:#555; font-size:0.95em;">Estimated Sale Price</p>
            <p class="result-price">${prediction:,.0f}</p>
        </div>
    """, unsafe_allow_html=True)

    with st.expander("🔍 See the inputs used for this prediction"):
        st.dataframe(input_row, use_container_width=True)

st.divider()
st.caption(
    "Built as part of the Sydney Housing Price Prediction and Decision Support System project. "
    "Model trained on manually collected sold-property data — treat estimates as indicative, not a formal valuation."
)