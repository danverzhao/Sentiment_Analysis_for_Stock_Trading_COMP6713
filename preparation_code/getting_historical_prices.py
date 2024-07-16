import yfinance as yf
import pandas as pd

# Define the stock symbol
symbols = ["AAPL"]

# Define the start and end dates
start_date = "2020-03-01"
end_date = "2020-08-01"


for symbol in symbols:
    # Create a Ticker object
    ticker = yf.Ticker(symbol)

    # Get the historical stock prices
    history = ticker.history(start=start_date, end=end_date)

    history.to_csv(f'{symbol}_prices_{start_date}_to_{end_date}.csv')