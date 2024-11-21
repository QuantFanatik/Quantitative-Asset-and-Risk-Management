import pandas as pd
import os

root = os.path.dirname(__file__)
returns_path = os.path.join(root, 'data', 'portfolio_returns.csv')
weights_path = os.path.join(root, 'data', 'portfolio_weights.csv')

portfolio_returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
portfolio_weights = pd.read_csv(weights_path, index_col=0, parse_dates=True, header=[0, 1]) 

print(portfolio_returns[portfolio_returns.index.year >= 2006].head(10))
print(portfolio_weights[portfolio_weights.index.year >= 2006].head(10))
print(portfolio_weights.columns.get_level_values(0).unique())

portfolio_weights = portfolio_weights.loc[:, ('equity_amer', slice(None))]
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 7))
plt.stackplot(portfolio_weights.index, portfolio_weights.T, labels=portfolio_weights.columns)
plt.title('ERC Portfolio Weights Over Time (Stacked Area)')
plt.xlabel('Date')
plt.ylabel('Weight')
plt.legend(title='ERC Components', loc='upper left')
plt.grid(True)
plt.show()