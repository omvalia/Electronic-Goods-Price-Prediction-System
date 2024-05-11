import streamlit as st
import pickle
import numpy as np

# Load your model and data
with open('pipeline_mobile.pkl', 'rb') as model_file, open('df_mobile.pkl', 'rb') as df_file:
    pipe = pickle.load(model_file)
    df = pickle.load(df_file)

# Set the title and page layout
st.title("Mobile Price Predictor")
st.markdown("---")

# Sidebar section for user input with style
st.sidebar.header("Configure Your Mobile")
st.sidebar.markdown("---")

# Brand
brand = st.sidebar.selectbox('Brand', df['Brand'].unique())

# Battery Capacity
battery_capacity = st.sidebar.number_input('Battery Capacity (mAh)')

# Screen Size
screen_size = st.sidebar.number_input('Screen Size (inches)')

# Touchscreen
touchscreen = st.sidebar.selectbox('Touchscreen', ['No', 'Yes'])

# RAM
ram = st.sidebar.number_input('RAM (MB)')

# Internal Storage
internal_storage = st.sidebar.number_input('Internal Storage (GB)')

# Rear Camera
rear_camera = st.sidebar.number_input('Rear Camera (MP)')

# Front Camera
front_camera = st.sidebar.number_input('Front Camera (MP)')

# Operating System
os = st.sidebar.selectbox('Operating System', df['Operating system'].unique())

# Wi-Fi
wifi = st.sidebar.selectbox('Wi-Fi', ['No', 'Yes'])

# Bluetooth
bluetooth = st.sidebar.selectbox('Bluetooth', ['No', 'Yes'])

# GPS
gps = st.sidebar.selectbox('GPS', ['No', 'Yes'])

# Number of SIMs
num_sims = st.sidebar.number_input('Number of SIMs')

# 3G
_3g = st.sidebar.selectbox('3G', ['No', 'Yes'])

# 4G/LTE
lte = st.sidebar.selectbox('4G/LTE', ['No', 'Yes'])

if st.sidebar.button('Predict Price'):
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    query = np.array([brand, battery_capacity, screen_size, touchscreen, ram, internal_storage, rear_camera, front_camera, os, wifi, bluetooth, gps, num_sims, _3g, lte], dtype=object)
    query = query.reshape(1, -1)

    predicted_price = np.exp(pipe.predict(query)[0])

    # Display the predicted price below the title
    st.markdown("---")
    st.header("Predicted Price")
    st.success(f"The predicted price of this configuration is ₹{predicted_price:.2f}")
