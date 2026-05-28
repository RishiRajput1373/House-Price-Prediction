# House-Price-Prediction

Machine Learning web application for predicting house prices using **Python**, **Pandas**, **Scikit-learn**, and **Streamlit**.

## Project Structure

```text
House-Price-Prediction/
├── Data/
│   └── house_data.csv
├── Model/
│   ├── train_model.py
│   ├── house_price_model.joblib      # generated after training
│   ├── scaler.joblib                 # generated after training
│   └── metrics.joblib                # generated after training
├── .streamlit/
│   └── config.toml
├── app.py
├── requirements.txt
└── README.md
```

## Features

- Data preprocessing and feature engineering
- `StandardScaler` for feature scaling
- `RandomForestRegressor` model training
- Model/scaler persistence using `joblib`
- Evaluation metrics: **R2 Score** and **MSE**
- Interactive Streamlit interface for house detail inputs:
  - bedrooms, bathrooms, sqft living, lot size, floors, grade, condition
  - zipcode, latitude, longitude
  - nearby house information (`sqft_living15`, `sqft_lot15`)

## Setup

```bash
pip install -r requirements.txt
```

## Train Model

```bash
python Model/train_model.py
```

This saves:
- `Model/house_price_model.joblib`
- `Model/scaler.joblib`
- `Model/metrics.joblib`

## Run Streamlit App

```bash
streamlit run app.py
```

The app auto-trains once if model artifacts are not present.

## Deployment

The repository is deployment-ready for Streamlit platforms (Streamlit Community Cloud or container-based hosting):
- Entry point: `app.py`
- Port/headless config: `.streamlit/config.toml`
- Dependencies: `requirements.txt`
