import pandas as pd
import numpy as np
import boto3
from io import BytesIO
import streamlit as st
import altair as alt
import snowflake.connector
import forecasting_functions as ff
import plotly.express as px
import plotly.graph_objs as go

# Layout of the main page
st.set_page_config(layout="wide")

color_map = {
    'Actual': '#4c95d9', 
    'Forecast': '#ff6a6a'   
}

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

df_prod = fetch_data(r"SELECT DATE, VALUE, PRODUCT FROM ATC1")
df_prod_scope = fetch_data(r"SELECT DATE, VALUE, PRODUCT, SCOPE FROM ATC1_BY_MARKET")
df_family = fetch_data(r"SELECT DATE, VALUE, PRODUCT FROM ATC2")
df_family_scope = fetch_data(r"SELECT DATE, VALUE, PRODUCT, SCOPE FROM ATC2_BY_MARKET")
for i in [df_prod, df_prod_scope, df_family, df_family_scope]:
    i['DATE'] = pd.to_datetime(i['DATE'])
    i['TYPE'] = 'Actual'

# Import the csv files from S3 bucket - CIP product table
bucket_name = "pharma-sales-forecasting"
file_key = "Product_base.csv"
df_product = load_data_from_s3(bucket_name, file_key)

# Extract the list of unique product
product_list = list(df_product['ATC_Class'].unique())
product_list.sort()
product_list = product_list[:5] + ["METFORMINE", "VITAMINES"]
family_list = list(df_product['ATC_Class2'].unique())
family_list.sort()
family_list = family_list[:5]

# Interface Streamlit
st.title("🏥 French Pharmaceutical Sales Forecasting")

st.sidebar.write("""This web application, made with Streamlit, is a personal project I undertook to practice with Time-series Forecasting. 
                 The technical stack used implies AWS, Snowflake, SQL, and Python. 
                 The aim of this application is to provide a trend analysis and trend prediction of the drug consumption in France.""")

tab1, tab2, tab3 = st.tabs(["Forecast by category", "Forecast by product", "Forecast by reference"])
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selection = st.selectbox('Product family to forecast:', family_list, key=1)    
    with col2:
        scope = st.selectbox('Forecasting scope:', ['Both','Community pharmacy', 'Hospital'], key=2)
    with col3:
        method = st.selectbox('Forecasting method:', ['Linear Regression', 'Moving Average', 'Exponential Smoothing', 'ARIMA', 'LSTM', 'Prophet'], key=3)
    with col4:
        prediction_timeframe = st.slider('Forecasting horizon (in months):', min_value=3, value=6, max_value=12, step=1, key=4)
    # Filter the dataframe
    if scope == 'Both':
        df = df_family[df_family['PRODUCT'] == selection]
    else:
        df = df_family_scope[(df_family_scope['PRODUCT'] == selection) & (df_family_scope['SCOPE'] == scope)]

with tab2:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selection = st.selectbox('Product to forecast:', product_list, key=5)    
    with col2:
        scope = st.selectbox('Forecasting scope:', ['Both','Community pharmacy', 'Hospital'], key=6)
    with col3:
        method = st.selectbox('Forecasting method:', ['Linear Regression', 'Moving Average', 'Exponential Smoothing', 'ARIMA', 'LSTM', 'Prophet'], key=7)
    with col4:
        prediction_timeframe = st.slider('Forecasting horizon (in months):', min_value=3, value=6, max_value=12, step=1, key=8)
    # Filter the dataframe
    if scope == 'Both':
        df = df_prod[df_prod['PRODUCT'] == selection]
    else:
        df = df_prod_scope[(df_prod_scope['PRODUCT'] == selection) & (df_prod_scope['SCOPE'] == scope)]

    if method == 'Linear Regression':
        predictions, curve = ff.predict_linear_regression(df, prediction_timeframe)
        text = "Linear regression is a forecasting methodology that predicts the value of a variable based on the linear relationship between that variable and one or more predictor variables. The method involves finding the best-fit line through the data points, which minimizes the sum of the squared differences between the observed values and the values predicted by the line."
    elif method == 'Moving Average':
        predictions = ff.predict_linear_regression(df, prediction_timeframe)
        # predictions = ff.predict_moving_average(df, prediction_timeframe)
    elif method == 'Exponential Smoothing':
        predictions = ff.predict_exponential_smoothing(df, prediction_timeframe)
    elif method == 'ARIMA':
        predictions = ff.predict_auto_arima(df, prediction_timeframe)
    elif method == 'LSTM':
        predictions = ff.predict_lstm(df, prediction_timeframe)
    elif method == 'Prophet':
        predictions = ff.predict_linear_regression(df, prediction_timeframe)
        # predicti ons = ff.predict_prophet(df, prediction_timeframe)

    with st.expander("Forecasting method explanations"):
        st.write(text)

    # Chart
    fig = px.line(predictions, x="DATE", y="VALUE", color="TYPE", color_discrete_map=color_map)
    if method == 'Linear Regression':
        new_trace = go.Scatter(x=curve['DATE'], y=curve['VALUE'], mode='markers+lines', name='Regression line', line=dict(color='black', dash='dot'), opacity=0.5)
        fig.add_trace(new_trace)
    fig.update_layout(legend=dict(yanchor="top",y=1.0,xanchor="right",x=1.0,bgcolor="rgba(255, 255, 255, 0.5)", borderwidth=1))
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

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