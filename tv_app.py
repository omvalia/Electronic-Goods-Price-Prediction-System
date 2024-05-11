import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load your model and data
with open('pipeline_tv.pkl', 'rb') as model_file, open('df_tv.pkl', 'rb') as df_file:
    pipe = pickle.load(model_file)
    df = pickle.load(df_file)

# Set the title and page layout
st.title("TV Price Predictor")
st.markdown("---")

# Sidebar section for user input with style
st.sidebar.header("Configure Your TV")
st.sidebar.markdown("---")

# Brand
brand = st.sidebar.selectbox('Brand', df['Brand'].unique())

# Resolution
resolution = st.sidebar.selectbox('Resolution', df['Resolution'].unique())

# Size
size = st.sidebar.number_input('Size (inches)')

# Operating System
os = st.sidebar.selectbox('Operating System', df['Operating System'].unique())

if st.sidebar.button('Predict Price'):
    query = pd.DataFrame({'Brand': [brand], 'Resolution': [resolution], 'Size': [size], 'Operating System': [os]})
    
    predicted_price = np.exp(pipe.predict(query)[0])

    # Display the predicted price below the title
    st.markdown("---")
    st.header("Predicted Price")
    st.success(f"The predicted price of this configuration is ₹{predicted_price:.2f}")
