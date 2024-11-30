import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import Bounds, LinearConstraint, minimize
import scipy.sparse.linalg as sparla
import cvxpy as cp
import os
import yfinance as yf
import itertools, sys
import sys
import threading
import time
from itertools import cycle

root = os.getcwd()
staticPath = os.path.join(root, 'data', 'Static.xlsx')
ritPath = os.path.join(root, 'data', 'DS_RI_T_USD_M.xlsx')
mvPath = os.path.join(root, 'data', 'DS_MV_USD_M.xlsx')
rfPath = os.path.join(root, 'data', 'Risk_Free_Rate.xlsx')


global ANNUALIZATION_FACTOR
ANNUALIZATION_FACTOR = 12

def excel_loader(path):
    data = pd.read_excel(path, usecols=lambda x: x != 'NAME', index_col=0).transpose()
    data.index = pd.to_datetime(data.index, format='%Y')
    data.index = data.index + pd.offsets.YearEnd()
    data.index.rename('DATE', inplace=True)
    data = data[data.index.year > 2004]
    nan_columns = data.iloc[0].loc[data.iloc[0].isna()].index
    data.loc['2005-12-31', nan_columns] = data.loc['2006-12-31', nan_columns]
    data.interpolate(method='linear', axis=0, inplace=True)

    return data

df = excel_loader(ritPath)


staticData = pd.read_excel(staticPath, engine='openpyxl')
masterData = pd.read_excel(ritPath, usecols=lambda x: x != 'NAME', index_col=0, engine='openpyxl').transpose()
masterData.index.rename('DATE', inplace=True) # print(sum(masterData.isna().any())) # Prices have no missing values
masterData = masterData[masterData.index.year > 2000]

capData = pd.read_excel(mvPath, usecols=lambda x: x != 'NAME', index_col=0, engine='openpyxl').transpose()
capData.index = pd.to_datetime(capData.index, format='%Y-%m-%d')
capData.index.rename('DATE', inplace=True)
capData = capData[capData.index.year > 2000] * 1e6

global masterIndex, global_tickers
masterIndex = masterData.index
global_tickers = list(masterData.columns)
df_dict = {}
mv_dict = {}
for region in ['AMER', 'EM', 'EUR', 'PAC']:
    filter = staticData['ISIN'][staticData['Region'] == region]
    df_dict[region] = masterData[filter].pct_change()
    mv_dict[region] = capData[filter]



print(df)



if choice == "Performance":
    st.title("Performance")

    list_portfolios = [
        'equity_amer', 'equity_em', 'equity_eur', 'equity_pac',
        'metals', 'commodities', 'volatilities', 'crypto', 'erc'
    ]
    selected_portfolios = []

    # Portfolio Selection
    st.subheader("Select portfolios to display", divider="gray")
    rows = [list_portfolios[i:i + 4] for i in range(0, len(list_portfolios), 4)]
    for row in rows:
        cols = st.columns(len(row))
        for col, portfolio in zip(cols, row):
            with col:
                toggle = st.checkbox(portfolio, value=False)
                if toggle:
                    selected_portfolios.append(portfolio)

    if st.button("Display All"):
        selected_portfolios = list_portfolios

    if selected_portfolios:
        # Load portfolio returns
        portfolio_returns = load_portfolio_returns()
        gamma_value = st.session_state.get('gamma_value', None)
        if gamma_value is None:
            st.warning("Please set your Gamma in the 'Risk Profiling' section.")
            st.stop()

        # Get returns for the selected gamma
        try:
            returns_gamma = portfolio_returns.xs(gamma_value, level='gamma')
        except KeyError:
            st.error(f"No data available for gamma value {gamma_value}")
            st.stop()

        # Compute log returns
        log_returns_gamma = np.log1p(returns_gamma)

        # Calculate cumulative returns for selected portfolios
        cumulative_returns_selected = log_returns_gamma[selected_portfolios].cumsum() + 1
        cumulative_returns_selected = cumulative_returns_selected[
            cumulative_returns_selected.index >= '2006-01-01'
        ]

        st.write("")
        st.subheader("Cumulative Returns and Drawdown", divider="gray")

        # Plot cumulative returns
        st.write("#### Cumulative Returns")
        st.line_chart(cumulative_returns_selected)

        # Calculate drawdown
        st.write("#### Drawdown")
        cumulative_returns_max = cumulative_returns_selected.cummax()
        drawdowns = (cumulative_returns_selected - cumulative_returns_max) / cumulative_returns_max
        st.line_chart(drawdowns)

        # Correlation Heatmap
        st.write("")
        st.subheader("Correlation Heatmap", divider="gray")
        # Compute the correlation matrix of the selected portfolios
        correlation_matrix = log_returns_gamma[selected_portfolios].corr()

        # Plot the heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            annot_kws={"color": "white"},
            xticklabels=correlation_matrix.columns,
            yticklabels=correlation_matrix.index,
            ax=ax
        )
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        ax.tick_params(axis='x', colors='white', labelrotation=45)
        ax.tick_params(axis='y', colors='white', labelrotation=0)
        st.pyplot(fig)

        # Calculate performance metrics
        st.write("")
        st.subheader("Summary Statistics", divider="gray")
        metrics_data = {}
        for portfolio in selected_portfolios:
            # Use log returns
            returns = log_returns_gamma[portfolio].dropna()
            cumulative_returns = returns.cumsum() + 1

            # Calculate mean return, volatility, Sharpe ratio
            mean_return = returns.mean() * 12  # Annualized
            volatility = returns.std() * np.sqrt(12)  # Annualized
            sharpe_ratio = mean_return / volatility if volatility != 0 else 0

            # Max drawdown calculation
            cumulative_returns_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns - cumulative_returns_max) / cumulative_returns_max
            max_drawdown = drawdown.min()

            # Drawdown duration calculations
            in_drawdown = drawdown < 0
            drawdown_periods = in_drawdown.astype(int).groupby((~in_drawdown).cumsum())
            drawdown_durations = drawdown_periods.apply(lambda x: (x.index[-1] - x.index[0]).days)

            # Compute max and average drawdown durations
            max_drawdown_duration = drawdown_durations.max()
            avg_drawdown_duration = drawdown_durations.mean()

            metrics_data[portfolio] = {
                "Mean Return": mean_return,
                "Volatility": volatility,
                "Sharpe Ratio": sharpe_ratio,
                "Max Drawdown": max_drawdown,
                "Max Drawdown<br>Duration (days)": max_drawdown_duration,
                "Average Drawdown<br>Duration (days)": avg_drawdown_duration,
            }

        # Create DataFrame for metrics
        metrics_df = pd.DataFrame.from_dict(metrics_data, orient='index')

        # Remove potential empty rows caused by missing or NaN data
        metrics_df = metrics_df.dropna(how='all')  # Drop rows where all elements are NaN

        # Reset index to make 'Portfolio' a column
        metrics_df = metrics_df.reset_index().rename(columns={'index': 'Portfolio'})

        # Define the highlight function
        def highlight_metrics_column(s):
            better_high = {
                "Mean Return": True,
                "Volatility": False,
                "Sharpe Ratio": True,
                "Max Drawdown": False,
                "Max Drawdown<br>Duration (days)": False,
                "Average Drawdown<br>Duration (days)": False,
            }
            is_better_high = better_high.get(s.name, True)
            min_val = s.min()
            max_val = s.max()

            range_adjustment = (max_val - min_val) * 0.1
            min_val -= range_adjustment
            max_val += range_adjustment

            s = s.fillna(min_val).replace([np.inf, -np.inf], min_val)

            if min_val == max_val:
                normalized = s * 0.0
            else:
                normalized = (s - min_val) / (max_val - min_val)

            if not is_better_high:
                normalized = 1 - normalized

            normalized = normalized.clip(0, 1)

            try:
                colors = [
                    f"background-color: rgba({255 - int(255 * x)}, {int(255 * x)}, 0, 0.8)"
                    for x in normalized
                ]
            except ValueError as e:
                st.warning(f"ValueError in color generation for column '{s.name}': {e}")
                st.write(f"Problematic values: {normalized}")
                colors = ["background-color: rgba(255, 255, 255, 0.8)" for _ in normalized]

            return colors

        # Create mapping for whether higher is better (excluding 'Portfolio')
        metrics_columns = metrics_df.columns.difference(['Portfolio'])
        styled_metrics = metrics_df.style.apply(
            highlight_metrics_column, subset=metrics_columns, axis=0
        ).format({
            "Mean Return": "{:.2%}",
            "Volatility": "{:.2%}",
            "Sharpe Ratio": "{:.2f}",
            "Max Drawdown": "{:.2%}",
            "Max Drawdown<br>Duration (days)": "{:.0f}",
            "Average Drawdown<br>Duration (days)": "{:.0f}",
        })

        # Hide the index to remove the number column
        styled_metrics = styled_metrics.hide(axis='index')

        # Add custom CSS for better UI
        custom_styles = """
        <style>
            .metrics-table th {
                position: sticky;
                top: 0;
                background-color: #1a1a1a;
                color: white;
                text-align: center;
            }
            .metrics-table td {
                text-align: center;
                padding: 8px;
            }
            .metrics-table tr:hover {
                background-color: #333333;
            }
            .metrics-table {
                border-collapse: collapse;
                width: 100%;
                margin: 0 auto;
            }
            .metrics-table td, .metrics-table th {
                border: 1px solid #555;
            }
        </style>
        """

        # Render the styled DataFrame to HTML without the index
        html = styled_metrics.to_html(classes="metrics-table", escape=False, index=False)

        # Add custom styles and table HTML
        st.markdown(custom_styles, unsafe_allow_html=True)
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.warning("Please select at least one portfolio to view its performance.")



