import os
import streamlit as st
import pandas as pd
import logging
from datetime import datetime
from src.fraud_models.inference import InferenceEngine

LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    print(f"Created directory: {LOG_DIR}")
log_filename = os.path.join(LOG_DIR, f"fraud_{datetime.now().strftime('%Y-%m-%d')}.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_filename), 
        logging.StreamHandler()            
    ]
)

logger = logging.getLogger(__name__)
# MODEL CACHING 
@st.cache_resource
def load_engine():
    # This ensures the Autoencoder, Scaler, and Encoder load ONLY ONCE
    return InferenceEngine(model_dir='models/autoencoder/')

engine = load_engine()

# UI 
st.set_page_config(page_title="Fraud Detection Inference", layout="centered")

st.title("🛡️ Fraud Detection Model Inference")
st.info("Currently using the Autoencoder Anomaly Detection model.")

# Sensitivity Control
with st.sidebar:
    st.header("Detection Settings")
    multiplier = st.slider("Anomaly Sensitivity (Multiplier)", 1.0, 5.0, 2.0, 0.1)
    st.write("Higher multiplier = stricter definition of 'Fraud'.")

# Tabbed Interface: Manual Input vs CSV View
tab1, tab2 = st.tabs(["Single Transaction", "Live Log View"])

with tab1:
    st.subheader("Transaction Parameters")
    
    # Matching the 10 required features
    col1, col2 = st.columns(2)
    with col1:
        source = st.selectbox("Source", ["SEO", "Ads", "Direct"])
        browser = st.selectbox("Browser", ["Chrome", "Firefox", "Safari", "IE", "Opera"])
        sex = st.selectbox("Sex", ["M", "F"])
        purchase_value = st.number_input("Purchase Value", value=100.0)
        age = st.number_input("Age", value=25)
        
    with col2:
        time_since_signup = st.number_input("Time Since Signup", value=5000)
        hour_of_day = st.number_input("Hour of Day (0-23)", 0, 23, 12)
        day_of_week = st.number_input("Day of Week (0-6)", 0, 6, 1)
        device_count = st.number_input("Device Count", value=1)
        ip_count = st.number_input("IP Count", value=1)

    if st.button("Predict Fraud Status", type="primary"):
        input_data = {
            'source': source, 'browser': browser, 'sex': sex,
            'purchase_value': purchase_value, 'age': age,
            'time_since_signup': time_since_signup, 'hour_of_day': hour_of_day,
            'day_of_week': day_of_week, 'device_count': device_count, 'ip_count': ip_count
        }

        try:
            # Inject the UI multiplier into the engine
            engine.fraud_multiplier = multiplier
            
            # Run Inference
            result = engine.predict(input_data)
            
            # LOGGING 
            logging.info(f"Input: {input_data} | Result: {result}")

            # UI DISPLAY
            st.divider()
            if result['status'] == "Fraud":
                st.error(f"### Result: {result['status']}")
            elif result['status'] == "Suspicious":
                st.warning(f"### Result: {result['status']}")
            else:
                st.success(f"### Result: {result['status']}")

            # Visualizing the Reconstruction Error
            st.metric("Model Reconstruction Error (MSE)", f"{result['anomaly_score']:.6f}")
            st.progress(min(result['anomaly_score'] / (engine.threshold * 5), 1.0))
            st.caption(f"Threshold: {engine.threshold:.6f} | Fraud Boundary: {engine.threshold * multiplier:.6f}")

        except Exception as e:
            st.error(f"Inference Error: {e}")
            logging.error(f"Inference Error: {e}")

with tab2:
    st.subheader("Recent Activity (from app_inference.log)")
    try:
        with open("app_inference.log", "r") as f:
            log_data = f.readlines()
            # Show last 10 log entries
            for line in log_data[-10:]:
                st.text(line.strip())
    except FileNotFoundError:
        st.write("No logs generated yet.")

