import pandas as pd
import os

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

print(returns)
print(frontiers)
