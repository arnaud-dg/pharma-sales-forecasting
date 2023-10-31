import pandas as pd
import numpy as np
import boto3
from io import BytesIO
import streamlit as st
import altair as alt

# Layout of the main page
st.set_page_config(layout="wide")

# Initializing du client S3
s3 = boto3.client('s3')

# Importing functions
def fetch_data(SQL_query):
    # Connection to snowflake and cursor creation
    conn = snowflake.connector.connect(**st.secrets["snowflake"])
    cur = conn.cursor()
    cur.execute(SQL_query)
    # Loading Data into a DataFrame
    df = pd.DataFrame.from_records(iter(cur), columns=[x[0] for x in cur.description])
    # Close the connection
    cur.close()
    conn.close()
    return df

def load_data_from_s3(bucket_name, file_key):
    """Get a .csv file from a S3 bucket and transform it as a dataframe"""
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    content = response['Body'].read()
    df = pd.read_csv(BytesIO(content))
    return df
  
# Import the csv files from S3 bucket - CIP product table
bucket_name = "pharma-sales-forecasting"
file_key = "Product_base.csv"
df_product = load_data_from_s3(bucket_name, file_key)

# Extract the list of unique product
product_list = list(df_product['ATC_Class'].unique())
product_list.sort()
product_list = product_list[:5]

# Interface Streamlit
st.title("🏥 French Pharmaceutical Sales Forecasting")

st.sidebar.write("""This web application, made with Streamlit, is a personal project I undertook to practice with Time-series Forecasting. 
                 The technical stack used implies AWS, Snowflake, SQL, and Python. 
                 The aim of this application is to provide a trend analysis and trend prediction of the drug consumption in France.""")
# data_to_forecast = st.sidebar.radio("What kind of data do you wish to forecast",["***A drug family***", "***A product***", "***A reference (CIP code)***"])
prediction_timeframe = st.sidebar.slider('How many months do you wish to predict?', min_value=3, value=6, max_value=12, step=1)
selected_product = st.sidebar.selectbox('Which product would you like to forecast?', product_list)
forecasting_method = st.sidebar.selectbox('Which forecasting method would you like to apply', ['ARIMA', 'Prophet', 'LSTM', 'Exponential Smoothing', 'Linear Regression'])

# Get the data from snowflake
query = "SELECT * FROM ATC1 WHERE ATC_Class = {}".format(selected_product)
st.write(query)
df_chart = fetch_data(query)
st.dataframe(df_chart)