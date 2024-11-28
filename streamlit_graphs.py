import streamlit as st
import plotly.express as px
import pandas as pd
import os

# Load data
def load_chunks(directory, base_filename):
    chunk_files = sorted(
        [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith(base_filename) and f.endswith('.csv')])
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found for base filename '{base_filename}' in '{directory}'")
    all_chunks = [pd.read_csv(chunk, parse_dates=["date"]) for chunk in chunk_files]
    return pd.concat(all_chunks, axis=0)

root = os.path.dirname(__file__)

returns = load_chunks(os.path.join(root, 'data'), 'portfolio_returns_gamma')
returns.set_index(["gamma", "date"], inplace=True)

frontiers = load_chunks(os.path.join(root, 'data'), 'efficient_frontiers_gamma')
frontiers.set_index(["gamma", "date", "portfolio"], inplace=True)

# @st.cache_data
def slice_data(data: pd.DataFrame, gammas=None, dates: pd.DatetimeIndex = None, assets=None, portfolios=None) -> pd.DataFrame:
    gammas = [gammas] if isinstance(gammas, (int, float)) else gammas
    portfolios = [portfolios] if isinstance(portfolios, str) else portfolios

    gamma_slice = data.index.get_level_values("gamma").intersection(gammas) if gammas is not None else slice(None)
    dates_slice = data.index.get_level_values("date").intersection(dates) if dates is not None else slice(None)

    if data.index.nlevels == 2:
        print("User warning: returns have no asset level, ignoring asset slice") if assets is not None else None
        portf_slice = data.columns.intersection(portfolios) if portfolios is not None else slice(None)
        return data.loc[(gamma_slice, dates_slice), portf_slice]
    
    if data.index.nlevels == 3:
        portf_slice = data.index.get_level_values("portfolio").intersection(portfolios) if portfolios is not None else slice(None)
        asset_slice = data.columns.intersection(assets) if assets is not None else slice(None)
        return data.loc[(gamma_slice, dates_slice, portf_slice), asset_slice]

allowable = {"gamma": list(frontiers.index.get_level_values(0).unique()), 
             "dates_frontier": pd.to_datetime(frontiers.index.get_level_values(1).unique()),
             "dates_returns": pd.to_datetime(returns.index.get_level_values(1).unique()),
             "portfolio":list(frontiers.index.get_level_values(2).unique()),
             }


# Example slicing
gamma_slice = allowable["gamma"][120:]
dates_slice = pd.date_range("2012-12-31", "2014-12-31")
portfolio_slice = allowable["portfolio"][1:4]

data = slice_data(returns, gammas=gamma_slice, dates=dates_slice, portfolios=portfolio_slice)
print(data)

data = slice_data(returns, gammas=3, dates=pd.date_range("2012-12-31", "2014-12-31"), portfolios="equity_amer")
print(data)
# print(select_by_years(data, [2013, 2014]))

allowable["dates_frontier"] = [d.date() for d in pd.to_datetime(frontiers.index.get_level_values(1).unique())]
allowable["dates_returns"] = [d.date() for d in pd.to_datetime(returns.index.get_level_values(1).unique())]

# Streamlit App
st.title("Dynamic Data Slicing and Visualization")

# Returns Section
st.header("Returns Slicing")
gamma_returns = st.slider("Select Gamma Range (Returns)", min(allowable["gamma"]), max(allowable["gamma"]), (min(allowable["gamma"]), max(allowable["gamma"])))
date_returns = st.slider("Select Date Range (Returns)", min(allowable["dates_returns"]), max(allowable["dates_returns"]), (min(allowable["dates_returns"]), max(allowable["dates_returns"])))

returns_sliced = slice_data(
    returns,
    gammas=range(int(gamma_returns[0]), int(gamma_returns[1]) + 1),
    dates=pd.date_range(date_returns[0], date_returns[1]),
)

st.write("Sliced Returns Data")
st.dataframe(returns_sliced if not returns_sliced.empty else pd.DataFrame(columns=returns.columns))

# Frontiers Section
st.header("Frontiers Slicing")
gamma_frontiers = st.slider("Select Gamma Range (Frontiers)", min(allowable["gamma"]), max(allowable["gamma"]), (min(allowable["gamma"]), max(allowable["gamma"])))
date_frontiers = st.slider("Select Date Range (Frontiers)", min(allowable["dates_frontier"]), max(allowable["dates_frontier"]), (min(allowable["dates_frontier"]), max(allowable["dates_frontier"])))
portfolio_frontiers = st.multiselect("Select Portfolios (Frontiers)", allowable["portfolio"], allowable["portfolio"])

frontiers_sliced = slice_data(
    frontiers,
    gammas=range(int(gamma_frontiers[0]), int(gamma_frontiers[1]) + 1),
    dates=pd.date_range(date_frontiers[0], date_frontiers[1]),
    portfolios=portfolio_frontiers,
)

st.write("Sliced Frontiers Data")
st.dataframe(frontiers_sliced if not frontiers_sliced.empty else pd.DataFrame(columns=frontiers.columns))