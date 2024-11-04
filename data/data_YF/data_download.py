import yfinance as yf
import pandas as pd

# Define the expanded universe of tickers for commodities and volatility indices
tickers = {
    "volatility_indices": ["^VIX", "^VXN", "^RVX", "^VXO", "^VVIX", "^MOVE"],  # VIX, Nasdaq VXN, Russell 2000 RVX, VXO, VVIX, MOVE Index
    "crypto": ["ETH-USD", "BTC-USD"],                                         # Ethereum, Bitcoin
    "commodities": ["GC=F", "CL=F", "SI=F", "HG=F", "PL=F", "PA=F",           # Gold, Crude Oil, Silver, Copper, Platinum, Palladium
                    "NG=F", "ZC=F", "ZS=F", "ZW=F", "LE=F", "HE=F"],          # Natural Gas, Corn, Soybeans, Wheat, Live Cattle, Lean Hogs
}

# Initialize a dictionary to store the data for each security type
data = {}

# Loop through each category and download the data for each ticker
for category, symbols in tickers.items():
    category_data = {}
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        # Download the maximum length daily data
        df = ticker.history(period="max", interval="1d")
        category_data[symbol] = df
    data[category] = category_data

# Access data, for example VIX data
vix_data = data["volatility_indices"]["^VIX"]

# Optionally, save each dataset to a CSV file
for category, category_data in data.items():
    for symbol, df in category_data.items():
        df.to_csv(f"/Users/davidhuber/Desktop/{symbol}_daily_data.csv")
