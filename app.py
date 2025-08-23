# Dependencies:
# streamlit, pandas, numpy, joblib

import streamlit as st
import pandas as pd
from joblib import load

st.set_page_config(page_title="Titanic Survival Prediction", page_icon=":ship:", layout="wide")
st.title('Titanic Survival Prediction App')
st.markdown("""
This app predicts whether a passenger would survive the Titanic disaster 
based on their personal and ticket details using a trained model.
""")
st.sidebar.header('Passenger Input Features')

# Load the trained model pipeline (should handle preprocessing internally)
model = load("./Titanic_Model.joblib")

# Default values as per your request
pclass = st.selectbox('Passenger Class', [1, 2, 3], index=0)  # default: 1
sex = st.selectbox('Sex', ['male', 'female'], index=0)        # default: 'male'
age = st.number_input('Age', min_value=0.0, max_value=100.0, value=28.0)      # default: 28.0
fare = st.number_input('Fare', min_value=0.0, max_value=600.0, value=99.50)   # default: 99.50
embarked = st.selectbox('Embarked', ['S', 'C', 'Q'], index=0) # default: 'S'
family_size = st.selectbox('Family Size', ['Alone', 'Small_Family', 'Large_Family'], index=0) # default: 'Alone'

predict = st.button('Predict Survival')

if predict:
    # Arrange input data in DataFrame matching model expectation
    input_data = pd.DataFrame([{
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'Fare': fare,
        'Embarked': embarked,
        'Family_Size': family_size
    }])

    # Model predicts survival (assuming 1=Survived, 0=Not Survived)
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("Survived!")
    else:
        st.error("Did not survive.")
