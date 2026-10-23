"""Streamlit interface for used-car resale price estimation."""
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("car_price_model.pkl")
CURRENT_FEATURES = ["Present_Price", "Driven_kms", "Owner", "Car_Age", "Fuel_Type", "Selling_type", "Transmission"]


@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("car_price_model.pkl is missing. Ensure the model file is included in deployment, or run 'python car.py' from the project root.")
    artifact = joblib.load(MODEL_PATH)
    if isinstance(artifact, dict) and "model" in artifact:
        bundle = artifact
    else:
        bundle = {"model": artifact, "reference_year": None, "legacy_model": True}
    model = bundle["model"]
    if not hasattr(model, "predict"):
        raise ValueError("car_price_model.pkl does not contain a usable prediction model. Retrain it with car.py.")
    bundle.setdefault("legacy_model", False)
    bundle.setdefault("reference_year", None)
    return bundle


def expected_feature_names(model):
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return list(names)
    for step in getattr(model, "named_steps", {}).values():
        names = getattr(step, "feature_names_in_", None)
        if names is not None:
            return list(names)
    return None


st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")
st.title("🚗 Car Price Prediction")
st.caption("Estimate used-car resale price from vehicle details. Results are model estimates, not guaranteed market valuations.")

try:
    bundle = load_model_bundle()
    model = bundle["model"]
except (FileNotFoundError, ValueError, OSError, EOFError, AttributeError, ImportError) as exc:
    st.error(f"Unable to load the saved car-price model: {exc}")
    st.info("Retrain the model using the repository's car.py, then redeploy the generated car_price_model.pkl.")
    st.stop()

reference_year = bundle.get("reference_year")
if reference_year:
    st.caption(f"Training car-age reference year: {reference_year}")
elif bundle.get("legacy_model"):
    st.warning("A legacy model file was detected. The app will check its input schema before predicting.")

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
        "Present_Price": [present_price], "Driven_kms": [driven_kms], "Owner": [owner],
        "Car_Age": [car_age], "Fuel_Type": [fuel_type], "Selling_type": [selling_type],
        "Transmission": [transmission],
    })
    expected = expected_feature_names(model)
    if expected and "Year" in expected and "Car_Age" not in expected:
        features["Year"] = (reference_year or date.today().year) - features["Car_Age"]
        features = features.drop(columns=["Car_Age"])

    if expected:
        missing = [column for column in expected if column not in features.columns]
        extra = [column for column in features.columns if column not in expected]
        if missing or extra:
            st.error(f"The saved model expects a different input schema (missing: {missing}; extra: {extra}). Retrain with the current car.py and redeploy car_price_model.pkl.")
            st.stop()
        features = features[expected]

    try:
        prediction = float(model.predict(features)[0])
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        st.error(f"Prediction failed because the saved model is incompatible with the current inputs: {exc}")
        st.info("Retrain with the current car.py and redeploy the newly generated car_price_model.pkl.")
        st.stop()

    st.success(f"Estimated Selling Price: ₹ {max(0.0, prediction):.2f} Lakhs")
