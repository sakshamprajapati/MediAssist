import streamlit as st
import pandas as pd
import numpy as np
import pickle
from collections import Counter
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="MediAssist", page_icon="🩺", layout="wide")

# Load saved models
svm_model = pickle.load(open('models/svm_model.pkl', 'rb'))
nb_model = pickle.load(open('models/nb_model.pkl', 'rb'))
rf_model = pickle.load(open('models/rf_model.pkl', 'rb'))

# Load data files
description_data = pd.read_csv("dataset/symptom_Description.csv")
precaution_data = pd.read_csv("dataset/symptom_precaution.csv")
doctor_data = pd.read_csv("dataset/doctors.csv")

# Load training data for column names
data = pd.read_csv('dataset/training_data.csv').dropna(axis=1)
symptoms = data.columns[:-1].values

# Label Encoder for 'prognosis' column
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

# Create data dictionary
symptom_index = { " ".join([i.capitalize() for i in value.split("_")]): index for index, value in enumerate(symptoms) }
data_dict = {
	"symptom_index": symptom_index,
	"predictions_classes": encoder.classes_
}

# Helper functions
def custom_mode(lst):
	counts = Counter(lst)
	max_count = max(counts.values())
	modes = [item for item, count in counts.items() if count == max_count]
	return modes[0]

def get_precautions(disease):
	precautions = precaution_data[precaution_data['Disease'] == disease].iloc[:, 1:].values.tolist()
	return precautions[0] if len(precautions) > 0 else ["No precautions available"]

def predictDisease(symptoms):
	symptoms = symptoms.split(",")
	input_data = [0] * len(data_dict["symptom_index"])
	for symptom in symptoms:
			index = data_dict["symptom_index"][symptom]
			input_data[index] = 1
	input_data = np.array(input_data).reshape(1, -1)

	rf_prediction = data_dict["predictions_classes"][rf_model.predict(input_data)[0]]
	nb_prediction = data_dict["predictions_classes"][nb_model.predict(input_data)[0]]
	svm_prediction = data_dict["predictions_classes"][svm_model.predict(input_data)[0]]

	final_prediction = custom_mode([rf_prediction, nb_prediction, svm_prediction])
	disp = description_data[description_data['Disease'] == final_prediction]['Description'].values[0] if final_prediction in description_data["Disease"].unique() else "No description available"
	precautions = get_precautions(final_prediction)

	return final_prediction, disp, precautions

def find_doctors(predicted_disease):
	return doctor_data[doctor_data['Diseases'].str.contains(predicted_disease, case=False)]

# Custom CSS
st.markdown("""
    <style>
    /* General styling */
    body {
        background-color: #121212;
        color: #e0e0e0;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1f1f1f !important;
    }
    .css-1d391kg .css-1v3fvcr {
        color: #e0e0e0 !important;
    }
    .css-1d391kg .css-qrbaxs {
        color: #9e9e9e !important;
    }

    /* Main content styling */
    .css-10trblm {
        color: #bb86fc !important;
    }
    .stButton>button {
        background-color: #bb86fc;
        color: #121212;
        border: none;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #3700b3;
        color: #e0e0e0;
    }

    /* Headers and Subheaders */
    .css-1ethveo {
        color: #bb86fc !important;
    }
    .css-1g5bu53 {
        color: #bb86fc !important;
    }

    /* Markdown styling */
    .stMarkdown {
        color: #e0e0e0;
    }

    /* Multiselect styling */
    .stMultiSelect label {
        color: #9e9e9e !important;
    }
    .stMultiSelect input {
        background-color: #1f1f1f;
        color: #e0e0e0;
    }

    /* Expander styling */
    .css-1gx9x3 {
        background-color: #1f1f1f;
        color: #e0e0e0 !important;
    }

    /* Custom cards */
    .card {
        background-color: #1f1f1f;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        padding: 20px;
        margin-bottom: 20px;
        color: #e0e0e0;
    }
    .doctor-card {
        border: 1px solid #3c3c3c;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: #1f1f1f;
    }
    .doctor-card strong {
        color: #bb86fc;
    }

    /* Footer styling */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #1f1f1f;
        color: #9e9e9e;
        text-align: center;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)


# Sidebar
st.sidebar.title("MediAssist")
st.sidebar.markdown("This application utilizes machine learning models—SVM, Naive Bayes, and Random Forest—to predict diseases based on the symptoms provided by the user.")
st.sidebar.markdown("### Instructions")
st.sidebar.markdown("1. Enter your symptoms in the main panel.\n2. Get the predicted disease, description, and precautions.\n3. Optionally, find doctors available for consultation.")

# Main page
st.title("MediAssist 🩺")
st.write("### Your Health Assistant")
st.write("Please enter your symptoms to get a disease prediction, description, precautions, and available doctors for consultation.")

# Enter symptoms
st.header("Enter your symptoms")

selected_symptoms = st.multiselect(
    'Select symptoms:',
    options=list(symptom_index.keys()),
    help="Choose from the list of symptoms"
)

# Display the selected symptoms
if selected_symptoms:
    formatted_symptoms = [symptom.capitalize() for symptom in selected_symptoms]
    formatted_symptoms_str = ", ".join(formatted_symptoms)
    st.write('You selected:', formatted_symptoms_str)

    with st.spinner('Predicting the disease...'):
        disease, description, precautions = predictDisease(",".join(selected_symptoms))
        
    st.success(f"**Prediction: {disease}**")
    st.write(f"**Description:** {description}")

    st.subheader("Precautions:")
    with st.expander("Click to view precautions"):
        for pre in precautions:
            st.text(f"- {pre}")

    consult_doctor = st.radio("Do you want to consult a doctor?", ("No", "Yes"))

    if consult_doctor == "Yes":
        matching_doctors = find_doctors(disease)

        if not matching_doctors.empty:
            st.subheader("Doctors available for the predicted disease:")
            for idx, doctor in matching_doctors.iterrows():
                st.markdown(
                    f"""
                    <div class="doctor-card">
                        <strong>Doctor Name:</strong> {doctor['DoctorName']}<br>
                        <strong>Specialization:</strong> {doctor['Specialization']}<br>
                        <strong>Age:</strong> {doctor['Age']}<br>
                        <strong>Gender:</strong> {doctor['Gender']}<br>
                        <strong>Address:</strong> {doctor['Address']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.warning("No doctors available for the predicted disease.")
else:
    st.info("Please select symptoms to get a prediction.")

# Footer
st.markdown("""
    <div class="footer">
        <p>This application is an ML model and should not be used as a substitute for professional medical advice. Always consult with a real doctor for accurate diagnosis and treatment.</p>
    </div>
""", unsafe_allow_html=True)
