
import streamlit as st
import pandas as pd
import joblib

model= joblib.load('House_price_prediction_pkl')

st.set_page_config(
   page_title="House Price Prediction",
   page_icon="🏠",
   layout="wide",
)

st.title("House Price Prediction App")
st.write("Enter the details below for prediction:")
st.divider()

income = st.number_input('Avg. Area Income',value=65000.0)
house_age = st.number_input('Avg. Area house Age', value = 4.0)
num_rooms = st.number_input('Avg. Area Number of Rooms',value = 7.0)
num_bedrooms= st.number_input('Avg. Area Number of Bedrooms',value = 4.0)
population= st.number_input('Area Population',value = 35000.0)

st.write(" ")
if st.button("Predict House Price"):

    features = pd.DataFrame(
        [[income, house_age, num_rooms, num_bedrooms, population]],
        columns=[
            "Avg. Area Income",
            "Avg. Area House Age",
            "Avg. Area Number of Rooms",
            "Avg. Area Number of Bedrooms",
            "Area Population"
        ]
    )

    prediction = model.predict(features)

    st.success(f"The Predicted House Price is : {prediction[0]:.2f}")

    if prediction[0] > 1000000:
        st.balloons()