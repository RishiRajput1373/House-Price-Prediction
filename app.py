import joblib
import pandas as pd
import streamlit as st

from Model.train_model import BASE_FEATURES, METRICS_PATH, MODEL_PATH, SCALER_PATH, engineer_features, train_and_save

st.set_page_config(page_title="House Price Prediction", page_icon="🏠")
st.title("🏠 House Price Prediction")
st.write("Enter house details to estimate the property price.")

if not MODEL_PATH.exists() or not SCALER_PATH.exists() or not METRICS_PATH.exists():
    st.info("Training model for first-time setup. Please wait...")
    with st.spinner("Training model..."):
        train_and_save()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
metrics = joblib.load(METRICS_PATH)

st.sidebar.header("Model Metrics")
st.sidebar.metric("R2 Score", f"{metrics['r2_score']:.4f}")
st.sidebar.metric("MSE", f"{metrics['mse']:.2f}")

col1, col2 = st.columns(2)
with col1:
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
    sqft_living = st.number_input("Sqft Living", min_value=200, max_value=20000, value=1800)
    sqft_lot = st.number_input("Lot Size (sqft)", min_value=300, max_value=100000, value=5000)
    floors = st.number_input("Floors", min_value=1.0, max_value=4.0, value=1.0, step=0.5)
    grade = st.number_input("Grade", min_value=1, max_value=13, value=7)

with col2:
    condition = st.number_input("Condition", min_value=1, max_value=5, value=3)
    zipcode = st.number_input("Zipcode", min_value=10000, max_value=99999, value=98001)
    lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=47.5, format="%.6f")
    longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-122.2, format="%.6f")
    sqft_living15 = st.number_input("Nearby Sqft Living", min_value=200, max_value=20000, value=1700)
    sqft_lot15 = st.number_input("Nearby Lot Size (sqft)", min_value=300, max_value=100000, value=4500)

if st.button("Predict Price"):
    input_df = pd.DataFrame([
        {
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "sqft_living": sqft_living,
            "sqft_lot": sqft_lot,
            "floors": floors,
            "grade": grade,
            "condition": condition,
            "zipcode": zipcode,
            "lat": lat,
            "long": longitude,
            "sqft_living15": sqft_living15,
            "sqft_lot15": sqft_lot15,
        }
    ])[BASE_FEATURES]

    transformed = engineer_features(input_df)
    transformed = transformed[metrics["feature_order"]]
    scaled = scaler.transform(transformed)
    prediction = model.predict(scaled)[0]

    st.success(f"Estimated House Price: ${prediction:,.2f}")
