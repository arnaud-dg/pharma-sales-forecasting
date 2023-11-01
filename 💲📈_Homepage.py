import pandas as pd
import numpy as np
import boto3
from io import BytesIO
import streamlit as st
import altair as alt
import snowflake.connector
import forecasting_functions as ff
import plotly.express as px

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

def load_data_from_snowflake():
    """Get a .csv file from a S3 bucket and transform it as a dataframe"""

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

tab1, tab2, tab3 = st.tabs(["Forecast by category", "Forecast by product", "Forecast by reference"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selection = st.selectbox('Product category to forecast:', product_list)
        query = r"SELECT DATE, VALUE FROM ATC1 WHERE PRODUCT = '{}'".format(selection)
    with col2:
        scope = st.selectbox('Forecasting scope:', ['Both','Community pharmacy', 'Hospital'])
        if scope == "Both":
            query = r"SELECT DATE, VALUE FROM ATC1 WHERE PRODUCT = '{}'".format(selection)
        else:
            query = r"SELECT DATE, VALUE FROM ATC1_BY_MARKET WHERE PRODUCT = '{}' AND MARKET = '{}'".format(selection, scope)
    with col3:
        method = st.selectbox('Forecasting method:', ['Linear Regression', 'Moving average', 'Exponential Smoothing', 'ARIMA', 'LSTM', 'Prophet'])
    with col4:
        prediction_timeframe = st.slider('Forecasting horizon (in months):', min_value=3, value=6, max_value=12, step=1)
    # Get the data from snowflake
    df = fetch_data(query)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['TYPE'] = 'Actual'
    # Prediction function
    if method == 'Linear Regression':
        predictions = ff.predict_linear_regression(df, prediction_timeframe)
    elif method == 'Exponential Smoothing':
        predictions = ff.predict_exponential_smoothing(df, prediction_timeframe)
    # Chart
    fig = px.line(predictions,x="DATE",y="VALUE",color="TYPE")
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

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