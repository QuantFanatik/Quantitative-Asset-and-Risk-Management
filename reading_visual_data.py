import pandas as pd
import numpy as np
import os

def load_chunks(directory, base_filename):
    chunk_files = sorted(
        [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith(base_filename) and f.endswith('.csv')])
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found for base filename '{base_filename}' in '{directory}'")
    all_chunks = [pd.read_csv(chunk, header=[0, 1], index_col=[0, 1, 2]) for chunk in chunk_files]
    return pd.concat(all_chunks, axis=0)

root = os.path.dirname(__file__)
returns_path = os.path.join(root, 'data', 'portfolio_returns.csv')
weights_path = os.path.join(root, 'data', 'portfolio_weights.csv')
frontiers_path = os.path.join(root, 'data', 'efficient_frontiers.csv')

portfolio_returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
portfolio_weights = pd.read_csv(weights_path, index_col=0, parse_dates=True, header=[0, 1])

# portfolio_frontiers = load_chunks(os.path.join(root, 'data'), 'efficient_frontiers')
portfolio_frontiers = pd.read_csv(frontiers_path, index_col=[0, 1, 2], header=[0, 1])
new_columns = [(top, "" if "Unnamed" in bottom else bottom) for top, bottom in portfolio_frontiers.columns]
portfolio_frontiers.columns = pd.MultiIndex.from_tuples(new_columns, names=["category", "attribute"])

# Examples usage
print(portfolio_returns[portfolio_returns.index.year >= 2010].head(2))
print(portfolio_weights[portfolio_weights.index.year >= 2009].head(2))
print(portfolio_weights.columns.get_level_values(0).unique())

# Select the weights of only the american equity portfolio
portfolio_weights = portfolio_weights.loc[:, ('equity_amer', slice(None))]

# The efficient frontiers are stored in a multi-index DataFrame, 
# the index levels are year, gamma, and portfolio (in that order)
# Example 1 - See unique portfolio names
print(portfolio_frontiers.index.get_level_values("portfolio").unique())

# Example 2 - Select the efficient frontier for the metals portfolio in year 3
year = 3
gamma = slice(None) # All gamma values
portfolio = "erc"

metals_data = portfolio_frontiers.loc[
    (year, gamma, portfolio), # Selecting rows
    (slice(None), ['expected_return', 'expected_variance', ])]  # Selecting columns

print(metals_data)

# Example 3 - Extract the gamma and expected return vectors
gamma_vector = metals_data.index.get_level_values('gamma').to_numpy()
expected_return_vector = metals_data[('metrics', 'expected_return')].to_numpy()

print("Gamma Vector:", gamma_vector)
print("Expected Return Vector:", expected_return_vector)

