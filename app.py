
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import base64


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Rwanda House Price Predictor",
    layout="wide"
)


def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded_image = base64.b64encode(file.read()).decode()

    st.markdown(
        f"""
        <style>

        .stApp {{
            background-image:
                linear-gradient(
                    rgba(0, 0, 0, 0.55),
                    rgba(0, 0, 0, 0.55)
                ),
                url("data:image/jpeg;base64,{encoded_image}");

            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .block-container {{
            background-color: rgba(0, 0, 0, 0.55);
            padding: 2rem 3rem;
            border-radius: 15px;
        }}

        h1, h2, h3, p, label {{
            color: white !important;
        }}

        [data-testid="stWidgetLabel"] p {{
            color: white !important;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


set_background("house_background.jpg")


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("house_price_model.sav")


try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the trained model.")
    st.info(
        "Make sure house_price_model.sav is in the same folder as app.py."
    )
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("Rwanda House Price Prediction System")

st.write(
    """
    This application uses a Multiple Linear Regression model to estimate
    the price of a house in Rwanda based on its physical and location
    characteristics.
    """
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("Enter House Information")

col1, col2 = st.columns(2)

with col1:

    area = st.number_input(
        "Area (m²)",
        min_value=20.0,
        max_value=2000.0,
        value=180.0,
        step=1.0
    )

    bedrooms = st.number_input(
        "Number of Bedrooms",
        min_value=1,
        max_value=20,
        value=4,
        step=1
    )

    bathrooms = st.number_input(
        "Number of Bathrooms",
        min_value=1,
        max_value=20,
        value=3,
        step=1
    )

    house_age = st.number_input(
        "House Age (Years)",
        min_value=0,
        max_value=100,
        value=5,
        step=1
    )


with col2:

    distance = st.number_input(
        "Distance to City Centre (km)",
        min_value=0.0,
        max_value=100.0,
        value=4.0,
        step=0.5
    )

    parking = st.number_input(
        "Parking Spaces",
        min_value=0,
        max_value=20,
        value=2,
        step=1
    )

    neighborhood = st.selectbox(
        "Neighborhood",
        [
            "Gasabo",
            "Huye",
            "Kicukiro",
            "Kigali City",
            "Musanze",
            "Nyarugenge"
        ]
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "Predict House Price",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # Validate inputs
        if area <= 0:
            st.error("Area must be greater than zero.")
            st.stop()

        if bedrooms <= 0:
            st.error("Number of bedrooms must be greater than zero.")
            st.stop()

        if bathrooms <= 0:
            st.error("Number of bathrooms must be greater than zero.")
            st.stop()

        if house_age < 0:
            st.error("House age cannot be negative.")
            st.stop()

        if distance < 0:
            st.error("Distance cannot be negative.")
            st.stop()

        if parking < 0:
            st.error("Parking spaces cannot be negative.")
            st.stop()


        # Create input DataFrame
        new_house = pd.DataFrame({
            "Area_m2": [area],
            "Bedrooms": [bedrooms],
            "Bathrooms": [bathrooms],
            "House_Age_Years": [house_age],
            "Distance_to_City_km": [distance],
            "Parking_Spaces": [parking],
            "Neighborhood": [neighborhood]
        })


        # Make prediction
        prediction = model.predict(new_house)[0]


        # ====================================================
        # RESULT
        # ====================================================

        st.success("Prediction completed successfully!")

        st.subheader("Predicted House Price")

        st.metric(
            label="Estimated Price",
            value=f"{prediction:,.2f} Million RWF"
        )


        # ====================================================
        # INPUT SUMMARY
        # ====================================================

        st.subheader("House Information")

        summary = pd.DataFrame({
            "Feature": [
                "Area",
                "Bedrooms",
                "Bathrooms",
                "House Age",
                "Distance to City Centre",
                "Parking Spaces",
                "Neighborhood"
            ],

            "Value": [
                f"{area:.1f} m²",
                bedrooms,
                bathrooms,
                f"{house_age} years",
                f"{distance:.1f} km",
                parking,
                neighborhood
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # VISUALIZATION
        # ====================================================

        st.subheader("Price Visualization")

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.bar(
            ["Predicted Price"],
            [prediction]
        )

        ax.set_ylabel(
            "Price (Million RWF)"
        )

        ax.set_title(
            "Estimated House Price"
        )

        st.pyplot(fig)


        # ====================================================
        # DOWNLOADABLE REPORT
        # ====================================================

        report = pd.DataFrame({
            "Area_m2": [area],
            "Bedrooms": [bedrooms],
            "Bathrooms": [bathrooms],
            "House_Age_Years": [house_age],
            "Distance_to_City_km": [distance],
            "Parking_Spaces": [parking],
            "Neighborhood": [neighborhood],
            "Predicted_Price_Million_RWF": [prediction]
        })

        csv_data = report.to_csv(
            index=False
        )

        st.download_button(
            label="Download Prediction Report",
            data=csv_data,
            file_name="house_price_prediction.csv",
            mime="text/csv",
            use_container_width=True
        )


    except Exception as e:

        st.error(
            "An error occurred while making the prediction."
        )

        st.exception(e)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.subheader("About the Model")

st.write(
    """
    The prediction system was developed using Multiple Linear Regression.
    The model uses house area, bedrooms, bathrooms, house age,
    distance to the city centre, parking spaces, and neighborhood
    as explanatory variables.
    """
)

st.caption(
    "Prediction values are estimates produced by the trained machine-learning model."
)

