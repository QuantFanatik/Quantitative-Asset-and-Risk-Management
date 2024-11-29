import pandas as pd
import os
import numpy as np

def load_chunks(directory, base_filename):
    chunk_files = sorted(
        [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith(base_filename) and f.endswith('.csv')])
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found for base filename '{base_filename}' in '{directory}'")
    all_chunks = [pd.read_csv(chunk, parse_dates=True) for chunk in chunk_files]
    return pd.concat(all_chunks, axis=0)

root = os.path.dirname(__file__)

returns = load_chunks(os.path.join(root, 'data'), 'portfolio_returns_gamma')
returns.set_index(["gamma", "date"], inplace=True)

frontiers = load_chunks(os.path.join(root, 'data'), 'efficient_frontiers_gamma')
frontiers.set_index(["gamma", "date", "portfolio"], inplace=True)

rates = load_chunks(os.path.join(root, 'data'), 'rf_rate')

print(rates)

print(frontiers)



"""
import pandas as pd
import os
import numpy as np

def load_chunks(directory, base_filename):
    chunk_files = sorted(
        [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith(base_filename) and f.endswith('.csv')]
    )
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found for base filename '{base_filename}' in '{directory}'")
    all_chunks = [pd.read_csv(chunk, parse_dates=True) for chunk in chunk_files]
    return pd.concat(all_chunks, axis=0)

# Specify the directory for the data
root = os.getcwd()  # Replace with the correct directory if needed

# Load data
returns = load_chunks(os.path.join(root, 'data'), 'portfolio_returns_gamma')
returns.set_index(["gamma", "date"], inplace=True)

frontiers = load_chunks(os.path.join(root, 'data'), 'efficient_frontiers_gamma')
frontiers.set_index(["gamma", "date", "portfolio"], inplace=True)

# Sort the index to prevent PerformanceWarning
frontiers.sort_index(inplace=True)

# Filter data for gamma = -0.5
gamma_value = -0.5
if gamma_value in frontiers.index.get_level_values('gamma'):
    # Access volatilities data for the specific gamma value
    Wheights = frontiers.loc[(gamma_value, slice(None), 'crypto'), :]
    print(f"Volatilities for gamma = {gamma_value}:")
    print(Wheights.tail(50))
else:
    print(f"No data found for gamma = {gamma_value}")
"""

# print(returns.loc[(3, slice(None)), 'erc'])
# print(frontiers.loc[(slice(None), pd.DatetimeIndex('2012-01-01'), 'volatilities'), ['expected_return', 'expected_variance']])

# print(returns.iloc[[23, 24, 25]])

"""print((1 + returns.loc[(0.0366666666666667, slice(None))]).cumprod())
# print(returns.iloc[(23, slice(None))].)

# print(returns.describe())

df = pd.read_csv('/Users/ivankhalin/Documents/code/MA3/Quantitative-Asset-and-Risk-Management/data/portfolio_returns.csv')
df.dropna(inplace=True)
df.set_index('DATE', inplace=True)
print((1+df).cumprod()) 

print(np.linspace(-0.5, 2, int(250+1)))"""