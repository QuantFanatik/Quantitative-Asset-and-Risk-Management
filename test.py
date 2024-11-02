print("bonjour")

# git add .
# git commit -m "new addition"
# git push origin main

import pandas as pd
import numpy as np

# Sample parameters
years = [2021, 2022, 2023]
portfolios = ['portfolio_1', 'portfolio_2']
gamma_values = np.linspace(-0.5, 1.5, 11)

# Example function to simulate different assets per portfolio
def generate_ticker_list(portfolio, year):
    np.random.seed(hash(f"{portfolio}_{year}") % (2**32))
    num_assets = np.random.randint(3, 6)
    return [f"asset_{i}" for i in range(num_assets)]

# Dictionary to hold data for constructing DataFrame
data = {}

# Loop over each year and portfolio to create efficient frontier data
for year in years:
    for portfolio in portfolios:
        # Generate unique assets for each portfolio and year
        tickers = generate_ticker_list(portfolio, year)
        
        # Simulate efficient frontier metrics
        expected_returns = np.random.random(len(gamma_values))
        expected_variances = np.random.random(len(gamma_values))
        expected_sharpes = np.random.random(len(gamma_values))
        
        # Simulate weights for each asset in the portfolio's ticker list
        weights = np.random.random((len(gamma_values), len(tickers)))
        weights /= weights.sum(axis=1, keepdims=True)  # Normalize weights to sum to 1
        
        # Store the data
        for i, gamma in enumerate(gamma_values):
            data[(year, gamma, portfolio)] = [expected_returns[i], expected_variances[i], expected_sharpes[i], *weights[i]]

# Define MultiIndex for rows and columns
index = pd.MultiIndex.from_tuples(data.keys(), names=["year", "gamma", "portfolio"])
columns = pd.MultiIndex.from_tuples(
    [("metrics", "expected_return"), ("metrics", "expected_variance"), ("metrics", "expected_sharpe")] +
    [("weights", asset) for asset in set(asset for tickers in [generate_ticker_list(p, y) for p in portfolios for y in years] for asset in tickers)],
    names=["category", "attribute"]
)

# Convert data dictionary to DataFrame
df = pd.DataFrame(data.values(), index=index, columns=columns)

# Display the structure of the DataFrame
print(df)

# Save to HDF5
# df.to_hdf('efficient_frontiers.h5', key='frontier_data', mode='w')