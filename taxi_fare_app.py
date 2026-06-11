
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Uber NCR Intelligence System",
    page_icon="🚖",
    layout="wide"
)


fare_model = joblib.load("fare_model.pkl")
status_model = joblib.load("status_model.pkl")

vehicle_types = joblib.load(
    "vehicle_types.pkl"
)

locations = joblib.load(
    "locations.pkl"
)

payments = joblib.load(
    "payments.pkl"
)


st.title(
    "🚖 Uber NCR Intelligence System"
)

st.caption(
    "Fare Prediction & Ride Status Prediction"
)

st.divider()

menu = st.sidebar.radio(
    "Choose Service",
    [
        "Fare Prediction",
        "Ride Status Prediction"
    ]
)

col1, col2 = st.columns(2)

with col1:

    vehicle = st.selectbox(
        "Vehicle Type",
        vehicle_types
    )

    pickup = st.selectbox(
        "Pickup Location",
        locations
    )

    distance = st.slider(
        "Ride Distance (KM)",
        1,
        100,
        10
    )

with col2:

    drop = st.selectbox(
        "Drop Location",
        locations
    )

    payment = st.selectbox(
        "Payment Method",
        payments
    )

    ride_date = st.date_input(
        "Ride Date"
    )

ride_time = st.time_input(
    "Ride Time"
)


hour = ride_time.hour

weekday = ride_date.strftime(
    "%A"
)

is_weekend = (
    1 if ride_date.weekday() >= 5
    else 0
)

input_df = pd.DataFrame({

    "Vehicle Type":[vehicle],

    "Pickup Location":[pickup],

    "Drop Location":[drop],

    "Ride Distance":[distance],

    "Payment Method":[payment],

    "Hour":[hour],

    "Weekday":[weekday],

    "Is_Weekend":[is_weekend]

})

if menu == "Fare Prediction":

    if st.button(
        "Predict Fare"
    ):

        fare = fare_model.predict(
            input_df
        )[0]

        st.success(
            f"Estimated Fare: ₹{fare:,.2f}"
        )

if menu == "Ride Status Prediction":

    if st.button(
        "Predict Ride Status"
    ):

        status = status_model.predict(
            input_df
        )[0]

        st.success(
            f"Predicted Status: {status}"
        )

st.divider()

st.caption(
    "Built using Scikit-Learn, HistGradientBoosting & Streamlit"
)
