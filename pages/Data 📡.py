import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv('data/nasa_clean.csv')

st.title('Asteroid Data Exploration')

st.subheader('Data Overview')
st.write(df.head(10))

st.subheader('Dataset Information')
st.text(f'Dataset Shape: {df.shape}')


hazardous_counts = df['Hazardous'].value_counts()
fig, ax = plt.subplots()
ax.pie(hazardous_counts, labels=hazardous_counts.index, autopct='%1.1f%%', startangle=90, colors=sb.color_palette("pastel"))
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Hazardous Classification')
st.pyplot(fig)
