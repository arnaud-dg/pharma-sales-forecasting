import pandas as pd
import numpy as np
import boto3
from io import BytesIO
import streamlit as st
import altair as alt
import snowflake.connector

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
product_list = product_list[:5] + ["METFORMINE", "VITAMINES"]

# Interface Streamlit
st.title("🏥 French Pharmaceutical Sales Forecasting")

st.sidebar.write("""This web application, made with Streamlit, is a personal project I undertook to practice with Time-series Forecasting. 
                 The technical stack used implies AWS, Snowflake, SQL, and Python. 
                 The aim of this application is to provide a trend analysis and trend prediction of the drug consumption in France.""")
# data_to_forecast = st.sidebar.radio("What kind of data do you wish to forecast",["***A drug family***", "***A product***", "***A reference (CIP code)***"])




tab1, tab2, tab3 = st.tabs(["Forecast by category", "Forecast by product", "Forecast by reference"])

with tab1:
    col1, col2, col3 = st.columns(4)
    with col1:
        selection = st.sidebar.selectbox('Product category to forecast:', product_list)
    with col2:
        scope = st.sidebar.selectbox('Forecasting scope:', ['Community pharmacy', 'Hospital', 'Both'])
    with col3:
        method = st.sidebar.selectbox('Forecasting method:', ['Linear Regression', 'Moving average', 'Exponential Smoothing', 'ARIMA', 'LSTM', 'Prophet'])
    with col4:
        prediction_timeframe = st.sidebar.slider('How many months do you wish to predict?', min_value=3, value=6, max_value=12, step=1)
    query = "SELECT * FROM ATC2 WHERE ATC_Class2 = 'VITAMINES'"
    df_chart = fetch_data(query)
    df_chart['SALESDATE'] = pd.to_datetime(df_chart['SALESDATE'])
    st.line_chart(data=df_chart, x='SALESDATE', y='NB_UNITS')


with tab2:
   st.header("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

with tab3:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

# Get the data from snowflake
# query = "SELECT * FROM ATC1 WHERE ATC_Class = '" + selected_product + "'"
# df_chart = fetch_data(query)
# df_chart['SALESDATE'] = pd.to_datetime(df_chart['SALESDATE'])

# query = "SELECT * FROM ATC2 WHERE ATC_Class2 = 'VITAMINES'"
# df_chart = fetch_data(query)
# df_chart['SALESDATE'] = pd.to_datetime(df_chart['SALESDATE'])

# # Affichage du graphique altair
# st.line_chart(data=df_chart, x='SALESDATE', y='NB_UNITS')
# st.dataframe(df_chart)