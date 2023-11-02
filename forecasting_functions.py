import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from prophet import Prophet

def predict_linear_regression(df, n_months):
    """
    Predict sales for the next 'n' months using linear regression.
    :param df: DataFrame containing a 'Value' column with monthly sales data.
    :n_months: Number of months to predict.
    :return: List with predictions for the next 'n' months.
    """
    # Create a column for the months in numerical form (1, 2, 3,...)
    df['month_num'] = range(1, len(df) + 1)
    # Split data into input variables X and target variable y
    X = df['month_num'].values.reshape(-1, 1)
    y = df['VALUE']

    # Find the last DATE occurrence in the DataFrame and create a list with the next 'n' months
    last_date = df['DATE'].iloc[-1]
    new_dates = pd.date_range(start=last_date, periods=n_months+1, freq='MS')[1:].to_list()

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X, y)

    # Regression curve
    regression_values = model.predict(X)
    regression_df = pd.DataFrame({'DATE': df['DATE'], 'VALUE': regression_values})
    regression_df.loc[regression_df['VALUE'] < 0, 'VALUE'] = 0
    # Prediction for the next n months
    future_months = np.array(range(len(df) + 1, len(df) + (n_months + 1))).reshape(-1, 1)
    predictions = model.predict(future_months)
    

    df = df.drop('month_num', axis=1)
    result = pd.DataFrame({'DATE': new_dates, 'VALUE': predictions})
    result['TYPE'] = 'Forecast'
    result.loc[result['VALUE'] < 0, 'VALUE'] = 0
    predictions = pd.concat([df, result], axis=0)

    return predictions, regression_df

def predict_exponential_smoothing(df, n_months):
        """
        Predict sales for the next 'n' months using exponential smoothing.
        :param df: DataFrame containing a 'VALUE' column with monthly sales data.
        :n_months: Number of months to predict.
        :return: DataFrame with predictions for the next 'n' months.
        """
        
        # Fit the exponential smoothing model
        model = SimpleExpSmoothing(df['VALUE']).fit()

        # Predict the next 'n' months
        forecasts = model.forecast(steps=n_months)
        
        # Find the last DATE occurrence in the DataFrame and create a list with the next 'n' months
        last_date = df['DATE'].iloc[-1]
        new_dates = pd.date_range(start=last_date, periods=n_months+1, freq='MS')[1:].to_list()

        result = pd.DataFrame({'DATE': new_dates, 'VALUE': forecasts})
        result['TYPE'] = 'Forecast'
        result.loc[result['VALUE'] < 0, 'VALUE'] = 0
        predictions = pd.concat([df, result], axis=0)
        
        return predictions
    
def predict_auto_arima(df, n_months):
        """
        Predict sales for the next 'n' months using auto ARIMA.
        
        :param df: DataFrame containing a 'VALUE' column with monthly sales data.
        :param n_months: Number of months to predict.
        :return: DataFrame with predictions for the next 'n' months.
        """
        
        # Fit the ARIMA model using auto_arima
        model = auto_arima(df['VALUE'], trace=True, error_action='ignore', suppress_warnings=True, seasonal=True)
        
        # Predict the next 'n' months
        forecasts = model.predict(n_periods=n_months)
        
        # Find the last DATE occurrence in the DataFrame and create a list with the next 'n' months
        last_date = df['DATE'].iloc[-1]
        new_dates = pd.date_range(start=last_date, periods=n_months+1, freq='MS')[1:].to_list()

        result = pd.DataFrame({'DATE': new_dates, 'VALUE': forecasts})
        result['TYPE'] = 'Forecast'
        result.loc[result['VALUE'] < 0, 'VALUE'] = 0
        predictions = pd.concat([df, result], axis=0)
        
        return predictions

def predict_lstm(df, n_months, n_input, n_features):
    """
    Predict sales for the next 'n' months using an LSTM network.
    :param df: DataFrame containing a 'Value' column with monthly sales data.
    :param n_months: Number of months to predict.
    :param n_input: The number of lag months to use for the LSTM input.
    :param n_features: The number of features to use in the LSTM model (usually 1 for univariate time series).
    :return: DataFrame with predictions for the next 'n' months.
    """
    # Ensure the data is in the correct shape
    df['VALUE'] = df['VALUE'].astype(float)

    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['VALUE'].values.reshape(-1, 1))

    # Prepare the data for LSTM
    generator = TimeseriesGenerator(scaled_data, scaled_data, length=n_input, batch_size=1)

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Fit the model
    model.fit(generator, epochs=100)

    # Make predictions
    last_n_months = scaled_data[-n_input:]
    predictions = []
    current_batch = last_n_months.reshape((1, n_input, n_features))

    for i in range(n_months):
        current_pred = model.predict(current_batch)[0]
        predictions.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

    # Inverse transform to get the predictions back to the original scale
    predictions = scaler.inverse_transform(predictions)

    # Prepare the dates for the forecast
    last_date = pd.to_datetime(df['DATE'].iloc[-1])
    prediction_dates = [last_date + pd.DateOffset(months=x) for x in range(1, n_months + 1)]

    # Create a DataFrame to hold the predictions
    forecast_df = pd.DataFrame(data=predictions, columns=['Forecast'])
    forecast_df['DATE'] = prediction_dates

    return forecast_df

def predict_prophet(df, n_months):
    """
    Predict sales for the next 'n' months using the Prophet algorithm.
    :param df: DataFrame containing 'ds' column with dates and 'y' column with monthly sales data.
    :param n_months: Number of months to predict.
    :return: DataFrame with the forecasted values for the next 'n' months.
    """
    # Make sure the DataFrame has the correct columns for Prophet
    df = df.rename(columns={'DATE': 'ds', 'VALUE': 'y'})

    # Initialize the Prophet model
    model = Prophet()
    
    # Fit the model with the historical data
    model.fit(df)

    # Create a DataFrame with future dates for prediction
    future = model.make_future_dataframe(periods=n_months, freq='M')

    # Use the model to make a forecast
    forecast = model.predict(future)

    # Extract the predicted values and dates
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n_months)

    # Rename the columns for consistency
    forecast = forecast.rename(columns={'ds': 'DATE', 'yhat': 'VALUE'})

    return forecast
