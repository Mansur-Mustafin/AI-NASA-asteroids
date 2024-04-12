import os
import pandas as pd
import numpy as np
import streamlit as st
from joblib import load

class Classifier:
    def __init__(self, path):
        self.model = load(path)

    def predict_hazardous(self, features):
        prediction = self.model.predict(features)
        return prediction[0]

# Load model filenames
model_files = [file for file in os.listdir('models/') if file.endswith('.joblib')]

# Streamlit webpage
st.title('Asteroid Classification App')

# Dropdown to select the model
selected_model_file = st.selectbox('Select a Model', ['Select a model...'] + model_files)

# Check if a model has been selected
if selected_model_file != 'Select a model...':
    classifier = Classifier(f'models/{selected_model_file}')

    # Getting user input
    st.subheader('Enter the asteroid characteristics:')
    absolute_magnitude = st.number_input('Absolute Magnitude', format="%.2f")
    est_dia_min = st.number_input('Estimated Diameter (min) in KM', format="%.2f")
    est_dia_max = st.number_input('Estimated Diameter (max) in KM', format="%.2f")
    close_approach_date = st.date_input('Close Approach Date')
    relative_velocity = st.number_input('Relative Velocity km per sec', format="%.2f")
    miss_distance = st.number_input('Miss Distance (kilometers)', format="%.2f")
    orbit_uncertainty = st.slider('Orbit Uncertainty', 0, 9, 5)
    minimum_orbit_intersection = st.number_input('Minimum Orbit Intersection', format="%.6f")

    # Button to make prediction
    if st.button('Predict Hazardous'):
        # Create an array with the input values
        input_data = np.array([[absolute_magnitude, est_dia_min, est_dia_max, close_approach_date.toordinal(),
                                relative_velocity, miss_distance, orbit_uncertainty, minimum_orbit_intersection]])
        # Convert input array to DataFrame
        input_df = pd.DataFrame(input_data, columns=['Absolute Magnitude', 'Est Dia in KM(min)', 
                                                     'Est Dia in KM(max)', 'Close Approach Date', 
                                                     'Relative Velocity km per sec', 'Miss Dist.(kilometers)', 
                                                     'Orbit Uncertainty', 'Minimum Orbit Intersection'])
        # Get the prediction
        result = classifier.predict_hazardous(input_df)
        if result == 1:
            st.success('The asteroid is potentially hazardous.')
        else:
            st.error('The asteroid is not hazardous.')
else:
    st.warning('Please select a model to proceed.')
