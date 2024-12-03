import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os


def get_path(filename):
    """Construct full path for the data file."""
    root = os.path.dirname(__file__)
    return os.path.join(root, 'data', filename)

def load_data(file, file_type='excel', sheet=0, cols=None, transpose=False):
    """Load data from Excel or CSV, setting index to 'Date' for CSV."""
    if file_type == 'excel':
        data = pd.read_excel(file, sheet_name=sheet, usecols=cols, index_col=0, engine='openpyxl')
    elif file_type == 'csv':
        data = pd.read_csv(file, index_col="Date", parse_dates=True)
    return data.transpose() if transpose else data

# Data Loading Functions
# ---------------------------------------------------------------------------------------
def load_chunks(directory, base_filename, parse_dates=None, date_column=None):
    chunk_files = sorted(
        [os.path.join(directory, f) for f in os.listdir(directory)
         if f.startswith(base_filename) and f.endswith('.csv')]
    )
    if not chunk_files:
        st.error(f"No chunk files found for base filename '{base_filename}' in '{directory}'")
        return pd.DataFrame()
    all_chunks = []
    for chunk in chunk_files:
        df = pd.read_csv(chunk, parse_dates=parse_dates)
        if date_column and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
        all_chunks.append(df)
    data = pd.concat(all_chunks, axis=0)
    return data

root = os.path.dirname(__file__)


def load_portfolio_returns():
    data = load_chunks(os.path.join(root, 'data'), 'portfolio_returns_gamma', parse_dates=["date"])
    data.set_index(["gamma", "date"], inplace=True)
    return data


def load_efficient_frontier_data():
    data = load_chunks(os.path.join(root, 'data'), 'efficient_frontiers_gamma', parse_dates=["date"])
    data.set_index(["gamma", "date", "portfolio"], inplace=True)
    data.sort_index(inplace=True)
    return data

def load_weights_data(sub_portfolio_list, sub_portfolio, gamma_value):
    data = load_chunks(os.path.join(root, 'data'), 'efficient_frontiers_gamma', parse_dates=["date"])
    data.set_index(["gamma", "date", "portfolio"], inplace=True)
    data = data.loc[(gamma_value, slice(None), sub_portfolio), :]
    return data[sub_portfolio_list]

def load_rates_data():
    data = load_chunks(os.path.join(root, 'data'), 'rf_rate')
    # Check for 'date' or 'Date' column
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
    elif 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        data.index.name = 'date'  # Rename the index to 'date'
    else:
        st.error("No 'date' or 'Date' column found in risk-free rate data.")
        return pd.DataFrame()
    # Rename 'RF' column to 'rf_rate' to match code elsewhere
    if 'RF' in data.columns:
        data.rename(columns={'RF': 'rf_rate'}, inplace=True)
    else:
        st.error("No 'RF' column found in risk-free rate data.")
        return pd.DataFrame()
    return data

# Load additional data if needed
# ---------------------------------------------------------------------------------------
dir_path = os.path.dirname(os.path.realpath(__file__))
master_path = os.path.join(dir_path, "data/data_YF/master_df.csv")
master_df = pd.read_csv(master_path, index_col=0, parse_dates=True)

# Help for making the web clean
# --------------------------------------------------------------------------------------
list_type_portfolio = ['equity_amer', 'equity_em', 'equity_eur', 'equity_pac',
                       'metals', 'commodities', 'crypto', 'volatilities', "erc"]

list_clean_name = ['Metals', 'Commodities', 'Crypto', 'Volatilities',
                   'North American Equities', 'Emerging Markets Equities', 'European Equities', 'Asia-Pacific Equities', "ERC"]

list_commodities = ["Lean_Hogs", "Crude_Oil", "Live_Cattle", "Soybeans", "Wheat", "Corn", "Natural_Gas"]
list_crypto = ["Bitcoin", "Ethereum"]
list_metals = ["Gold", "Platinum", "Palladium", "Silver", "Copper"]
list_volatilities = ["Russell_2000_RVX", "VVIX_VIX_of_VIX", "MOVE_bond_market_volatility",
                     "VXO-S&P_100_volatility", "Nasdaq_VXN", "VIX"]

list_ERC =['equity_amer', 'equity_em', 'equity_eur', 'equity_pac',
                       'metals', 'commodities', 'crypto', 'volatilities']

df_commodities = master_df[list_commodities].pct_change()
df_crypto = master_df[list_crypto].pct_change()
df_metals = master_df[list_metals].pct_change()
df_volatilities = master_df[list_volatilities].pct_change()



corr_matrix = master_df.corr()

list_data_equity_path = os.path.join(root, 'data', 'list_equity')
list_data_equity_amer = pd.read_csv(os.path.join(list_data_equity_path, "equity_amer.csv"))
list_data_equity_em = pd.read_csv(os.path.join(list_data_equity_path, "equity_em.csv"))
list_data_equity_eur = pd.read_csv(os.path.join(list_data_equity_path, "equity_eur.csv"))
list_data_equity_pac = pd.read_csv(os.path.join(list_data_equity_path, "equity_pac.csv"))

master_data_full = load_data(get_path('DS_RI_T_USD_M.xlsx'), cols=lambda x: x != 'NAME', transpose=True)

df_equity_amer = master_data_full[list_data_equity_amer]



df_equity_amer = master_data_full[list_data_equity_amer["ISIN"]]
df_equity_em = master_data_full[list_data_equity_em["ISIN"]]
df_equity_eur = master_data_full[list_data_equity_eur["ISIN"]]
df_equity_pac = master_data_full[list_data_equity_pac["ISIN"]]

df_equity_amer = df_equity_amer.pct_change()
df_equity_em = df_equity_em.pct_change()
df_equity_eur = df_equity_eur.pct_change()
df_equity_pac = df_equity_pac.pct_change()

print(df_equity_amer)
print(df_equity_em)
print(df_equity_eur)
print(df_equity_pac)







