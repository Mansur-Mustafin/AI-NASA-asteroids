import os
import pandas as pd
import numpy as np
import streamlit as st
from joblib import load
import datetime
import time

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

    absolute_magnitude = st.number_input('Absolute Magnitude', format="%.2f")
    est_dia_min = st.number_input('Estimated Diameter (min) in KM', format="%.2f")

    close_approach_date = st.date_input('Close Approach Date')
    close_approach_datetime = datetime.datetime.combine(close_approach_date, datetime.time())
    epoch_date_close_approach = int(close_approach_datetime.timestamp() * 1000)

    relative_velocity = st.number_input('Relative Velocity km per sec', format="%.2f")
    miss_distance = st.number_input('Miss Distance (kilometers)', format="%.2f")
    orbit_uncertainty = st.slider('Orbit Uncertainty', 0, 9, 5)
    minimum_orbit_intersection = st.number_input('Minimum Orbit Intersection', format="%.6f")
    eccentricity = st.number_input('Eccentricity', format="%.6f")
    semi_major_axis = st.number_input('Semi Major Axis', format="%.6f")
    inclination = st.number_input('Inclination', format="%.6f")
    asc_node_longitude = st.number_input('Asc Node Longitude', format="%.6f")
    perihelion_distance = st.number_input('Perihelion Distance', format="%.6f")
    perihelion_arg = st.number_input('Perihelion Arg', format="%.6f")
    perihelion_time = st.number_input('Perihelion Time', format="%.6f")
    mean_anomaly = st.number_input('Mean Anomaly', format="%.6f")
    mean_motion = st.number_input('Mean Motion', format="%.6f")

    # Button to make prediction
    if st.button('Predict Hazardous'):
        features = [
            absolute_magnitude,
            est_dia_min,
            epoch_date_close_approach,
            relative_velocity,
            miss_distance,
            orbit_uncertainty,
            minimum_orbit_intersection,
            eccentricity,
            semi_major_axis,
            inclination,
            asc_node_longitude,
            perihelion_distance,
            perihelion_arg,
            perihelion_time,
            mean_anomaly,
            mean_motion
        ]
        features_array = np.array([features])
        is_hazardous = classifier.predict_hazardous(features_array)

        if is_hazardous:
            st.success('The asteroid is predicted to be hazardous.')
        else:
            st.success('The asteroid is predicted to be not hazardous.')
else:
    st.warning('Please select a model to proceed.')
