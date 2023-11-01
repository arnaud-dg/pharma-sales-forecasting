import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

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
    new_dates = pd.date_range(start=last_date, periods=n_months+1, freq='M')[1:]  # On commence à [1:] pour exclure la dernière date déjà présente
    new_dates['TYPE'] = 'Forecast'

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X, y)

    # Prediction for the next n months
    future_months = np.array(range(len(df) + 1, len(df) + (n_months + 1))).reshape(-1, 1)
    predictions = model.predict(future_months)

    predictions = pd.concat([predictions, new_dates], axis=1)

    return predictions
