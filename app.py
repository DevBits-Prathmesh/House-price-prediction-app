import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title='ML_Model', page_icon='🕊️')

# Fixing Windows path input
def load_model(model_path):
    try:
        model_path = model_path.replace("\\", "/")
        if not os.path.exists(model_path):
            st.error(f'Model file not found at: {model_path}')
            return None
        return joblib.load(model_path)
    except Exception as e:
        st.error(f'Error loading model: {str(e)}')
        return None

# Load model with progress
with st.spinner('Loading prediction model...'):
    house_model = load_model(r"X:\Study\streamlit_apps\diabeties app\read_house.pkl")

# Check if model loaded successfully
if house_model is None:
    st.error("Failed to load the model. Cannot continue.")
    st.stop()

# UI Generation
st.title('🏠 HOUSE PRICE PREDICTION')
st.write('This will predict the price of a house according to your requirements.')
st.write('Let us know your requirements for a house:')

with st.form("house_form"):
    size = st.number_input('Size of house in square Foot:')
    room = st.number_input('Number of rooms:')
    bath = st.slider('Number of bathrooms:',max_value=10)
    garege = st.number_input('Garage Capacity:')
    age = st.number_input('Age of house:')
    garden = st.radio('Want garden?', ['YES', 'NO'])
    submit = st.form_submit_button('Predict House Price')

if submit:
    garden_value = 1 if garden == 'YES' else 0
    input_data = pd.DataFrame([{
        "Size_sqft": int(size),
        'Num_Rooms': int(room),
        'Num_Bathrooms': int(bath),
        'Garage_Capacity': int(garege),
        'Has_Garden': garden_value,
        'Age_of_House': int(age)
    }])

    try:
        with st.spinner('Processing...'):
            pred = house_model.predict(input_data)[0]
        st.success('## Prediction Result')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Predicted House Price', f'₹{pred:,.0f}')
        with col2:
            st.write("🏡 Based on your preferences!")
    except Exception as e:
        st.error(f'Prediction failed: {str(e)}')
