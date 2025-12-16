import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Load the Brains ---
@st.cache_resource # This makes it load fast
def load_assets():
    model = joblib.load('solar_model.pkl')
    le_utility = joblib.load('le_utility.pkl')
    le_county = joblib.load('le_county.pkl')
    le_developer = joblib.load('le_developer.pkl')
    return model, le_utility, le_county, le_developer

model, le_utility, le_county, le_developer = load_assets()

# --- 2. The User Interface (Frontend) ---
st.title('☀️ Solar Energy Predictor')
st.markdown("### Optimize your Solar Installation")
st.write("Enter your project details below to estimate annual energy production.")

# Create two columns for a neat layout
col1, col2 = st.columns(2)

with col1:
    # Dropdowns for Categorical Data (Using the loaded classes)
    selected_utility = st.selectbox("Select Utility Company", le_utility.classes_)
    selected_county = st.selectbox("Select County", le_county.classes_)
    selected_developer = st.selectbox("Select Developer", le_developer.classes_)

with col2:
    # Sliders/Inputs for Numerical Data
    system_size = st.number_input("System Size (kWdc)", min_value=1.0, max_value=5000.0, value=5.0)
    battery_size = st.number_input("Battery Size (kWac)", min_value=0.0, max_value=500.0, value=0.0)
    year = st.number_input("Installation Year", min_value=2000, max_value=2030, value=2025)
    month = st.slider("Installation Month", 1, 12, 6)

# --- 3. The Logic (Backend) ---
if st.button('Predict Energy Output'):
    try:
        # Encode the inputs (English -> Math)
        utility_encoded = le_utility.transform([selected_utility])[0]
        county_encoded = le_county.transform([selected_county])[0]
        
        # Handle new developers not seen in training
        try:
            developer_encoded = le_developer.transform([selected_developer])[0]
        except:
            developer_encoded = le_developer.transform(['Unknown'])[0] # Fallback

        # Prepare the row for the model [Utility, County, Developer, Size, Battery, Year, Month]
        # (Order must match EXACTLY how we trained X_train)
        input_data = np.array([[utility_encoded, county_encoded, developer_encoded, 
                                system_size, battery_size, year, month]])

        # Predict
        prediction = model.predict(input_data)
        
        # --- 4. Display Result ---
        st.success(f"Estimated Annual Production: **{prediction[0]:,.2f} kWh**")
        
        # Industry Bonus: Show the "Value"
        estimated_savings = prediction[0] * 0.20 # Assuming $0.20 per kWh
        st.info(f"💰 Potential Annual Savings: ${estimated_savings:,.2f} (at $0.20/kWh)")

    except Exception as e:
        st.error(f"Error: {e}")