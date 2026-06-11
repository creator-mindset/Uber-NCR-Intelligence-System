
import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Uber NCR Intelligence System",
    page_icon="🚖",
    layout="wide"
)

@st.cache_resource
def load_files():
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

    return (
        fare_model,
        status_model,
        vehicle_types,
        locations,
        payments
    )

(
    fare_model,
    status_model,
    vehicle_types,
    locations,
    payments
) = load_files()


st.title("🚖 Uber NCR Intelligence System")

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
        min_value=1,
        max_value=100,
        value=10
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

    "Vehicle Type": [vehicle],

    "Pickup Location": [pickup],

    "Drop Location": [drop],

    "Ride Distance": [distance],

    "Payment Method": [payment],

    "Hour": [hour],

    "Weekday": [weekday],

    "Is_Weekend": [is_weekend]

})

if menu == "Fare Prediction":

    st.subheader("💰 Fare Prediction")

    if st.button("Predict Fare"):

        fare = fare_model.predict(
            input_df
        )[0]

        st.metric(
            label="Estimated Fare",
            value=f"₹{fare:,.2f}"
        )


if menu == "Ride Status Prediction":

    st.subheader(
        "🚕 Ride Status Prediction"
    )

    if st.button(
        "Predict Ride Status"
    ):

        status = status_model.predict(
            input_df
        )[0]

        st.metric(
            label="Predicted Ride Status",
            value=status
        )

st.divider()

st.caption(
    "Built using Python, Scikit-Learn, Streamlit and Machine Learning"
)
```
