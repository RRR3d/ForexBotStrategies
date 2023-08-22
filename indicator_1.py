import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import time


def calculate_rsi(data, window):
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


rsi_window = 14
symbol = "EURUSD=X"

while True:
    current_time = datetime.datetime.now().time()
    if current_time.minute % 5 == 0:  # Fetch data every 5 minutes
        data = yf.download(symbol, period="1d", interval="5m")

        if not data.empty:
            data["RSI"] = calculate_rsi(data, rsi_window)

            # Indicator: Buy when RSI is below 30, Sell when RSI is above 70
            data["Signal"] = 0  # 1: Buy signal, -1: Sell signal
            data.loc[data["RSI"] < 30, "Signal"] = 1
            data.loc[data["RSI"] > 70, "Signal"] = -1

            latest_data = data.iloc[-1]
            print("Time:", latest_data.name)
            print("Close:", latest_data["Close"])
            print("RSI:", latest_data["RSI"])
            print("Signal:", latest_data["Signal"])
        else:
            print("No data available.")

    # Wait for the next 5-minute interval
    time.sleep(60 * 5)
