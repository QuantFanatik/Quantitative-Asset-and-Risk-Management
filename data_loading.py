import os
import pandas as pd

'''This file is used to load data from the data files and organize it into a single .csv file to be used in optimization.'''

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

static_data = load_data(get_path('Static.xlsx'))
master_data_full = load_data(get_path('DS_RI_T_USD_M.xlsx'), cols=lambda x: x != 'NAME', transpose=True)
cap_data = load_data(get_path('DS_MV_USD_M.xlsx'), cols=lambda x: x != 'NAME', transpose=True) * 1e6
non_equities = load_data(get_path('data_YF/master_df.csv'), file_type='csv')
rf_rate = pd.read_excel(get_path('Risk_Free_Rate.xlsx'), usecols=[1], engine='openpyxl') / 100
rf_index = pd.date_range("2000-01-31", "2024-01-31", freq='M')
rf_rate["date"] = rf_index
rf_rate.set_index("date", inplace=True)

equity_portfolios = {
    'equity_amer': ['AMER'],
    'equity_em': ['EM'],
    'equity_eur': ['EUR'],
    'equity_pac': ['PAC']
}

non_equity_portfolios = {
    'metals': ['Gold', 'Silver', 'Platinum', 'Palladium', 'Copper'],
    'commodities': ['Corn', 'Crude_Oil', 'Lean_Hogs', 'Live_Cattle', 'Natural_Gas', 'Soybeans', 'Wheat'],
    'crypto': ['Bitcoin', 'Ethereum'],
    'volatilities': ['VIX', 'MOVE_bond_market_volatility', 'VVIX_VIX_of_VIX', 'VXO-S&P_100_volatility', 'Nasdaq_VXN', 'Russell_2000_RVX']
}

# Organize equity prices by portfolio using ISINs from static data
equity_prices = {}
for region, ticker in equity_portfolios.items():
    isin_filter = static_data['ISIN'][static_data['Region'] == ticker[0]]
    equity_prices[region] = master_data_full[isin_filter]

# Organize non-equity prices by category using predefined tickers
non_equity_prices = {}
for category, tickers in non_equity_portfolios.items():
    if all(ticker in non_equities.columns for ticker in tickers):
        non_equity_prices[category] = non_equities[tickers]

# Concatenate equity and non-equity prices into a single DataFrame
all_prices = pd.concat(
    [pd.concat(equity_prices.values(), keys=equity_prices.keys(), axis=1),
     pd.concat(non_equity_prices.values(), keys=non_equity_prices.keys(), axis=1)],
    axis=1
)

all_prices = all_prices.reindex(master_data_full.index)
all_prices.index.name = 'DATE'
cap_data.index.name = 'DATE'

all_prices.to_csv(get_path('all_prices.csv'))
cap_data.to_csv(get_path('cap_data.csv'))
rf_rate.to_csv(get_path('rf_rate.csv'))