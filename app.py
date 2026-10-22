"""Streamlit interface for used-car resale price estimation."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("car_price_model.pkl")


@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "car_price_model.pkl is missing. Run 'python car.py' from the project root first."
        )
    artifact = joblib.load(MODEL_PATH)
    # Support older artifacts that stored the sklearn Pipeline directly.
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact
    return {"model": artifact, "reference_year": None}


st.title("🚗 Car Price Prediction")
st.caption("Estimate used-car resale price from vehicle details. Results are model estimates, not guaranteed market valuations.")

try:
    bundle = load_model_bundle()
    model = bundle["model"]
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

if bundle.get("reference_year"):
    st.caption(f"Training car-age reference year: {bundle['reference_year']}")

left, right = st.columns(2)
with left:
    present_price = st.number_input("Present Price (Lakhs)", min_value=0.0, value=5.0, step=0.1)
    driven_kms = st.number_input("Driven KMs", min_value=0, value=20000, step=1000)
    owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])
    car_age = st.number_input("Car Age (Years)", min_value=0, value=5, step=1)
with right:
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    selling_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

if st.button("Predict Price", type="primary", use_container_width=True):
    features = pd.DataFrame({
        "Present_Price": [present_price],
        "Driven_kms": [driven_kms],
        "Owner": [owner],
        "Car_Age": [car_age],
        "Fuel_Type": [fuel_type],
        "Selling_type": [selling_type],
        "Transmission": [transmission],
    })
    prediction = float(model.predict(features)[0])
    st.success(f"Estimated Selling Price: ₹ {max(0.0, prediction):.2f} Lakhs")
