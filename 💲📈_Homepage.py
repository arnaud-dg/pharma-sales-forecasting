import pandas as pd
import numpy as np
import boto3
from io import BytesIO
import streamlit as st
import altair as alt
from streamlit_extras.app_logo import add_logo
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras import extra

# Layout of the main page
st.set_page_config(layout="wide")

# Initializing du client S3
s3 = boto3.client('s3')

def load_data_from_s3(bucket_name, file_key):
    """Get a .csv file from a S3 bucket and transform it as a dataframe"""
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    content = response['Body'].read()
    df = pd.read_csv(BytesIO(content))
    return df
  
# Import the csv files from S3 bucket
# 1. Pharmaceutical sales (=FACT table including time series)
bucket_name = "pharma-sales-forecasting"
file_key = "French_pharmaceutical_sales.csv"
df = load_data_from_s3(bucket_name, file_key)

# Interface Streamlit
st.title("Pharmaceutical sales forecasting")

st.dataframe(df)




# Drop-down list of the sidebar
# df_disease = fetch_data("select $1 from available_diseases")
# st.sidebar.write("""This web application, made with Streamlit, is a personal project I undertook to practice with AWS, Snowflake, SQL, and Python.
# The aim of this application is to provide a synthetic analysis of past and ongoing clinical studies for a given pathology.
# To limit the data volume, only a few pathologies have been set up. The data comes from the Clinicaltrials.gov API.""")
# st.sidebar.write("""Enjoy the journey! :sunglasses:""")
# st.sidebar.markdown("""---""")
# selected_disease = st.sidebar.selectbox("Please select a pathology:", df_disease['$1'].tolist())

# st.title('🏥 Clinical Trials .Gov Explorer 🧑‍⚕️')
