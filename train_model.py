import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('Data/data.csv')

# Data preprocessing
data = data.dropna()
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data['sale_year'] = data['date'].dt.year
data['sale_month'] = data['date'].dt.month
data['sale_day'] = data['date'].dt.day
data = data.drop(columns=['id', 'date'])
data = data.dropna()

# Split data into features and target variable
X = data.drop('price', axis=1)
y = data['price']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, Y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the model
os.makedirs('Model', exist_ok=True)
joblib.dump(model, 'Model/house_price_model.pkl')
joblib.dump(scaler, 'Model/scaler.pkl')
