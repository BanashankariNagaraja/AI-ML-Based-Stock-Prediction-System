import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import os


def predict_stock(ticker):

    # DOWNLOAD STOCK DATA

    df = yf.download(ticker, period="6mo")

    if df.empty:
        return None

    # KEEP CLOSE PRICE

    df = df[['Close']]

    # CREATE FUTURE PREDICTION COLUMN

    df['Prediction'] = df[['Close']].shift(-2)

    # FEATURES

    X = np.array(df.drop(['Prediction'], axis=1))[:-2]

    # LABELS

    y = np.array(df['Prediction'])[:-2]

    # SPLIT DATA

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # CREATE MODEL

    model = LinearRegression()

    # MACHINE LEARNING TRAINING

    model.fit(X_train, y_train)

    # TEST PREDICTIONS

    predictions = model.predict(X_test)

    # ACCURACY

    accuracy = round(r2_score(y_test, predictions) * 100, 2)

    # FUTURE PREDICTION

    future_days = np.array(
        df.drop(['Prediction'], axis=1)
    )[-2:]

    future_prediction = model.predict(future_days)

    # CREATE CHART

    plt.figure(figsize=(10,5))

    plt.plot(df['Close'])

    plt.title(f'{ticker} Stock Price')

    plt.xlabel('Days')

    plt.ylabel('Price')

    os.makedirs('static/charts', exist_ok=True)

    chart_path = f'static/charts/{ticker}.png'

    plt.savefig(chart_path)

    plt.close()

    return {
        'prediction': future_prediction,
        'accuracy': accuracy,
        'chart': chart_path
    }