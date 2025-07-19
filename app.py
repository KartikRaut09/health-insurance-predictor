import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

MODEL_PATH = "insurance_model.pkl"

# Load dataset
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv'
    df = pd.read_csv(url)
    df_encoded = pd.get_dummies(df, drop_first=True)
    return df, df_encoded

# Train model and save
def train_model(data):
    X = data.drop('charges', axis=1)
    y = data['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = model.score(X_test, y_test)

    joblib.dump(model, MODEL_PATH)
    return model, mse, r2

# Load model or train if not found
df, df_encoded = load_data()
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    mse, r2 = None, None
else:
    model, mse, r2 = train_model(df_encoded)

# Streamlit UI
st.title("🏥 Health Insurance Cost Predictor")
st.markdown("Predict your estimated insurance charges based on personal details.")

if mse and r2:
    st.sidebar.subheader("📊 Model Performance")
    st.sidebar.write(f"**MSE:** {mse:,.2f}")
    st.sidebar.write(f"**R² Score:** {r2:.4f}")

# User inputs
age = st.slider("Age", 18, 65, 30)
sex = st.selectbox("Sex", ["Male", "Female"])
bmi = st.slider("BMI (Body Mass Index)", 10.0, 45.0, 25.0)
children = st.slider("Number of Children", 0, 5, 1)
smoker = st.selectbox("Do you smoke?", ["Yes", "No"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Encode input
sex_male = 1 if sex == "Male" else 0
smoker_yes = 1 if smoker == "Yes" else 0
region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0

input_data = pd.DataFrame({
    'age': [age],
    'bmi': [bmi],
    'children': [children],
    'sex_male': [sex_male],
    'smoker_yes': [smoker_yes],
    'region_northwest': [region_northwest],
    'region_southeast': [region_southeast],
    'region_southwest': [region_southwest]
})

# Log prediction
def log_prediction(data, prediction):
    log = data.copy()
    log['predicted_charge'] = prediction
    log.to_csv("user_predictions.csv", mode='a', index=False, header=not os.path.exists("user_predictions.csv"))

# Feature importance
def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    plt.figure(figsize=(8, 5))
    plt.barh(np.array(feature_names)[sorted_idx], importance[sorted_idx])
    plt.xlabel("Feature Importance")
    plt.title("XGBoost Feature Importance")
    st.pyplot(plt.gcf())

# Predict
if st.button("Predict Insurance Cost"):
    prediction = model.predict(input_data)
    st.success(f"💰 Estimated Insurance Cost: ₹{prediction[0]:,.2f}")
    log_prediction(input_data, prediction[0])

# Show feature importance
st.subheader("📈 Feature Importance")
plot_feature_importance(model, input_data.columns)