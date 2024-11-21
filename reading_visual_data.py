import pandas as pd
import numpy as np
import os

root = os.path.dirname(__file__)
returns_path = os.path.join(root, 'data', 'portfolio_returns.csv')
weights_path = os.path.join(root, 'data', 'portfolio_weights.csv')
frontiers_path = os.path.join(root, 'data', 'efficient_frontiers.csv')

portfolio_returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
portfolio_weights = pd.read_csv(weights_path, index_col=0, parse_dates=True, header=[0, 1])

portfolio_frontiers = pd.read_csv(frontiers_path, header=[0, 1], index_col=0)
new_columns = [(top, "" if "Unnamed" in bottom else bottom) for top, bottom in portfolio_frontiers.columns]
portfolio_frontiers.columns = pd.MultiIndex.from_tuples(new_columns, names=["category", "attribute"])
print(portfolio_frontiers)

# print(portfolio_returns[portfolio_returns.index.year >= 2006].head(10))
# print(portfolio_weights[portfolio_weights.index.year >= 2006].head(10))
# print(portfolio_weights.columns.get_level_values(0).unique())

# portfolio_weights = portfolio_weights.loc[:, ('equity_amer', slice(None))]
# import matplotlib.pyplot as plt
# plt.figure(figsize=(14, 7))
# plt.stackplot(portfolio_weights.index, portfolio_weights.T, labels=portfolio_weights.columns)
# plt.title('ERC Portfolio Weights Over Time (Stacked Area)')
# plt.xlabel('Date')
# plt.ylabel('Weight')
# plt.legend(title='ERC Components', loc='upper left')
# plt.grid(True)
# plt.show()

# df = pd.read_csv(frontiers_path, header=[0, 1], index_col=0)
# df.replace(np.nan, 0, inplace=True)
# df.columns.set_names(["category", "attribute"], inplace=True)
# print(df.head())

