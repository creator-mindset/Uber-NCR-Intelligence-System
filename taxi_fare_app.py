import streamlit as st
import pandas as pd
import joblib



fare_model = joblib.load("fare_model.pkl")
status_model = joblib.load("status_model.pkl")



st.set_page_config(
    page_title="Uber NCR Smart Ride Predictor",
    page_icon="🚖",
    layout="wide"
)

st.title("🚖 Uber NCR Smart Ride Predictor")


menu = st.sidebar.radio(
    "Choose Service",
    [
        "Booking Value Prediction",
        "Ride Status Prediction"
    ]
)



vehicle = st.selectbox(
    "Vehicle Type",
    [
        "Auto",
        "Bike",
        "Mini",
        "Prime Sedan",
        "Prime SUV"
    ]
)

pickup = st.text_input(
    "Pickup Location"
)

drop = st.text_input(
    "Drop Location"
)

distance = st.number_input(
    "Ride Distance (km)",
    min_value=0.1,
    value=5.0
)

driver_rating = st.slider(
    "Driver Rating",
    1.0,
    5.0,
    4.0
)

customer_rating = st.slider(
    "Customer Rating",
    1.0,
    5.0,
    4.0
)

payment = st.selectbox(
    "Payment Method",
    [
        "Cash",
        "UPI",
        "Card"
    ]
)

ride_date = st.date_input(
    "Ride Date"
)

ride_time = st.time_input(
    "Ride Time"
)



year = ride_date.year
month = ride_date.month
day = ride_date.day

weekday = ride_date.strftime("%A")

is_weekend = 1 if ride_date.weekday() >= 5 else 0

hour = ride_time.hour



input_df = pd.DataFrame({

    'Vehicle Type':[vehicle],
    'Pickup Location':[pickup],
    'Drop Location':[drop],

    'Ride Distance':[distance],

    'Driver Ratings':[driver_rating],
    'Customer Rating':[customer_rating],

    'Payment Method':[payment],

    'Hour':[hour],

    'Year':[year],
    'Month':[month],
    'Day':[day],

    'Weekday':[weekday],
    'Is_Weekend':[is_weekend]
})



if menu == "Booking Value Prediction":

    st.subheader("💰 Predict Booking Value")

    if st.button("Predict Fare"):

        prediction = fare_model.predict(
            input_df
        )

        st.success(
            f"Estimated Fare: ₹{prediction[0]:.2f}"
        )



if menu == "Ride Status Prediction":

    st.subheader("🚕 Predict Ride Status")

    if st.button("Predict Ride Status"):

        prediction = status_model.predict(
            input_df
        )

        st.success(
            f"Predicted Ride Status: {prediction[0]}"
        )
