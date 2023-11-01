import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def predict_linear_regression(df, n_months):
    """
    Predict sales for the next 'n' months using linear regression.
    :param df: DataFrame containing a 'sales' column with monthly sales data.
    :n_months: Number of months to predict.
    :return: List with predictions for the next 'n' months.
    """
    # Create a column for the months in numerical form (1, 2, 3,...)
    df['month_num'] = range(1, len(df) + 1)

    # Split data into input variables X and target variable y
    X = df['month_num'].values.reshape(-1, 1)
    y = df['sales']

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X, y)

    # Prediction for the next 6 months
    future_months = np.array(range(len(df) + 1, len(df) + 7)).reshape(-1, 1)
    predictions = model.predict(future_months)

    return predictions
