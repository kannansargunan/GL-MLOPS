import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Package Purchase Prediction")
st.write("""
This application predicts whether a customer is likely to purchase a tourism package 
based on their profile, demographics, and sales pitch interaction details.
Please enter the customer and pitch details below to get a prediction.
""")

# User input fields
age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2, step=1)
preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
number_of_trips = st.number_input("Number of Trips Annually", min_value=0, max_value=20, value=2, step=1)
passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
own_car = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
number_of_children_visiting = st.number_input("Number of Children Visiting (Under 5)", min_value=0, max_value=5, value=0, step=1)
designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", min_value=0.0, max_value=200000.0, value=20000.0, step=500.0)
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3, step=1)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, max_value=120.0, value=15.0, step=1.0)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': type_of_contact,
    'CityTier': city_tier,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': passport,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'ProductPitched': product_pitched,
    'NumberOfFollowups': number_of_followups,
    'DurationOfPitch': duration_of_pitch
}])

# Predict button
if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"The customer is **likely** to purchase the tourism package. (Probability: {probability:.2%})")
    else:
        st.warning(f"The customer is **unlikely** to purchase the tourism package. (Probability: {probability:.2%})")
