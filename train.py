import joblib
import pandas as pd
# from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# Load Boston Housing dataset (Alternative since sklearn removed load_boston in newer versions)
from sklearn.datasets import fetch_openml

def load_boston_data():
    """Fetch Boston Housing Data from OpenML"""
    boston = fetch_openml(name='boston', version=1, as_frame=True, parser='pandas')
    X = boston.data
    y = boston.target
    return X, y

# Load Data
X, y = load_boston_data()

# Select features used in app.py
selected_features = ["CHAS", "RM", "TAX", "PTRATIO", "B", "LSTAT"]
X = X[selected_features]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = GradientBoostingRegressor()
model.fit(X_train_scaled, y_train)

# Save the model
joblib.dump(model, "boston_housing_prediction.joblib")
joblib.dump(scaler, "scaler.joblib")

print("Model and scaler saved successfully!")
