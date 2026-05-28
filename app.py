import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('Model/house_price_model.pkl')
scaler = joblib.load('Model/scaler.pkl')
feature_order = [
    'bedrooms',
    'bathrooms',
    'sqft_living',
    'sqft_lot',
    'floors',
    'waterfront',
    'view',
    'condition',
    'grade',
    'sqft_above',
    'sqft_basement',
    'yr_built',
    'yr_renovated',
    'zipcode',
    'lat',
    'long',
    'sqft_living15',
    'sqft_lot15',
    'sale_year',
    'sale_month',
    'sale_day'
]

# Streamlit app
st.title('House Price Prediction')
# Input fields for user to enter house features
bedrooms = st.number_input('Number of Bedrooms', min_value=0, max_value=10, value=3)
bathrooms = st.number_input('Number of Bathrooms', min_value=0.0, max_value=10.0, value=2.0)
sqft_living = st.number_input('Square Footage of Living Area', min_value=0, max_value=10000, value=2000)
sqft_lot = st.number_input('Square Footage of Lot', min_value=0, max_value=100000, value=5000)
floors = st.number_input('Number of Floors', min_value=0.0, max_value=10.0, value=1.0)
waterfront = st.selectbox('Waterfront View', options=[0, 1])
view = st.selectbox('View Quality', options=[0, 1, 2, 3, 4])
condition = st.selectbox('Condition', options=[1, 2, 3,
4, 5])
grade = st.selectbox('Grade', options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
sqft_above = st.number_input('Square Footage Above Ground', min_value=0, max_value=10000, value=1500)
sqft_basement = st.number_input('Square Footage of Basement', min_value=0, max_value=5000, value=500)
yr_built = st.number_input('Year Built', min_value=1900, max_value=2026, value=2000)
yr_renovated = st.number_input('Year Renovated', min_value=0, max_value=2026, value=0)
zipcode = st.number_input('Zipcode', min_value=10000, max_value=99999, value=98052)
lat = st.number_input('Latitude', value=47.6062)
long = st.number_input('Longitude', value=122.3321)
sqft_living15 = st.number_input('Living Area of Nearest 15 Homes', min_value=0, max_value=10000, value=1800)
sqft_lot15 = st.number_input('Lot Size of Nearest 15 Homes', min_value=0, max_value=100000, value=6000)
sale_year = st.number_input('Sale Year', min_value=1900, max_value=2026, value=2014)
sale_month = st.number_input('Sale Month', min_value=1, max_value=12, value=6)
sale_day = st.number_input('Sale Day', min_value=1, max_value=31, value=15)

# Predict button
if st.button('Predict Price'):
    # Create a DataFrame with the exact feature names used during training
    input_row = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft_living': sqft_living,
        'sqft_lot': sqft_lot,
        'floors': floors,
        'waterfront': waterfront,
        'view': view,
        'condition': condition,
        'grade': grade,
        'sqft_above': sqft_above,
        'sqft_basement': sqft_basement,
        'yr_built': yr_built,
        'yr_renovated': yr_renovated,
        'zipcode': zipcode,
        'lat': lat,
        'long': long,
        'sqft_living15': sqft_living15,
        'sqft_lot15': sqft_lot15,
        'sale_year': sale_year,
        'sale_month': sale_month,
        'sale_day': sale_day,
    }
    input_data = pd.DataFrame([[input_row[column] for column in feature_order]], columns=feature_order)

    # Make prediction using the loaded model
    input_data_scaled = scaler.transform(input_data)
    predicted_price = model.predict(input_data_scaled)[0]
    st.subheader(f'Predicted House Price: ${predicted_price:,.2f}')

