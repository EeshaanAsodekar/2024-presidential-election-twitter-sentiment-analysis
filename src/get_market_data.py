import yfinance as yf
import pandas as pd

# Define the tickers for indexes and market variables
tickers = [
    '^RUA',     # Russell 3000
    '^GSPC',    # S&P 500
    '^NDX',     # NASDAQ-100
    'CL=F',     # Crude Oil (WTI)
    'NG=F',     # Natural Gas
    'DX-Y.NYB', # US Dollar Index (DXY)
    '^DJUSEN'   # Dow Jones U.S. Oil & Gas Index
]

# Define the date range
end_date = '2024-10-16'
start_date = '2024-10-01'

# Initialize an empty DataFrame to store the adjusted closing prices
closing_prices = pd.DataFrame()

# Fetch historical adjusted closing prices for each ticker
for ticker in tickers:
    try:
        # Download historical data from Yahoo Finance
        data = yf.download(ticker, start=start_date, end=end_date)

        # If data is found, add the 'Adj Close' prices to the DataFrame
        if not data.empty:
            closing_prices[ticker] = data['Adj Close']
        else:
            print(f"No data found for ticker: {ticker}")
    except Exception as e:
        # Handle errors during the data fetching process
        print(f"Error fetching data for ticker {ticker}: {e}")

print(closing_prices.head())

closing_prices.to_csv("data/raw/market_data.csv")
print("Market data saved to market_data.csv")
