import streamlit as st
import pickle
import joblib
import numpy as np

# Load your model and data
with open('pipe.pkl', 'rb') as pipe_file, open('df.pkl', 'rb') as df_file:
    pipe = pickle.load(pipe_file)
    df = joblib.load(df_file)

# Set the title and page layout
st.title("Laptop Price Predictor")
st.markdown("---")

# Sidebar section for user input with style
st.sidebar.header("Configure Your Laptop")
st.sidebar.markdown("---")

# Brand
company = st.sidebar.selectbox('Brand', df['Company'].unique())

# Type
laptop_type = st.sidebar.selectbox('Type', df['TypeName'].unique())

# Ram
ram = st.sidebar.selectbox('RAM (in GB)', df['Ram'].unique())

# Weight
weight = st.sidebar.number_input('Weight of the Laptop (in kg)')

# Touchscreen
touchscreen = st.sidebar.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.sidebar.selectbox('IPS Display', ['No', 'Yes'])

# Screen Size
screen_size = st.sidebar.number_input('Screen Size (in inches)')

# Screen Resolution
resolution = st.sidebar.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2561x1440', '2304x1440'])

# CPU Brand
cpu = st.sidebar.selectbox('CPU Brand', df['Cpu brand'].unique())

# HDD
hdd = st.sidebar.selectbox('HDD (in GB)', df['HDD'].unique())

# SSD
ssd = st.sidebar.selectbox('SSD (in GB)', df['SSD'].unique())

# GPU Brand
gpu = st.sidebar.selectbox('GPU Brand', df['Gpu brand'].unique())

# Operating System
os = st.sidebar.selectbox('Operating System', df['os'].unique())

if st.sidebar.button('Predict Price'):
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    x_res = int(resolution.split('x')[0])
    y_res = int(resolution.split('x')[1])
    ppi = ((x_res ** 2) + (y_res ** 2)) ** 0.5 / screen_size

    query = np.array([company, laptop_type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os], dtype=object)
    query = query.reshape(1, -1)

    predicted_price = np.exp(pipe.predict(query)[0])

    # Display the predicted price below the title
    st.markdown("---")
    st.header("Predicted Price")
    st.success(f"The predicted price of this configuration is ₹{predicted_price:.2f}")
