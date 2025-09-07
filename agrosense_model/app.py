import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="🌱 Smart Crop Recommendation", layout="centered")

# -------- Train Model --------
@st.cache_data
def train_model():
    # Load dataset
    data = pd.read_csv("data_core.csv")
    
    # Features & Target
    X = data[['Temparature', 'Humidity', 'Moisture']]
    y = data['Crop Type']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    return model, accuracy

# Load model
model, accuracy = train_model()

# -------- UI --------
st.title("🌱 Smart Crop Recommendation System")
st.write("Get the best crop suggestion based on **Temperature, Humidity, and Moisture**.")

# Sidebar Input
st.sidebar.header("Enter Soil & Weather Parameters")

temperature = st.sidebar.slider("Temperature (°C)", 0, 50, 25)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
moisture = st.sidebar.slider("Moisture (%)", 0, 100, 40)

# Prediction Button
if st.sidebar.button("Suggest Crop"):
    user_input = pd.DataFrame(
        [[temperature, humidity, moisture]],
        columns=['Temparature', 'Humidity', 'Moisture']
    )
    prediction = model.predict(user_input)[0]
    st.success(f"🌾 Recommended Crop: **{prediction}**")

# Show Model Accuracy
st.sidebar.markdown("---")
st.sidebar.write(f"✅ Model Accuracy: **{accuracy*100:.2f}%**")
