import pandas as pd
import os
import numpy as np


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

def load_chunks(directory, base_filename):
    chunk_files = sorted(
        [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith(base_filename) and f.endswith('.csv')])
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found for base filename '{base_filename}' in '{directory}'")
    all_chunks = [pd.read_csv(chunk, parse_dates=True) for chunk in chunk_files]
    return pd.concat(all_chunks, axis=0)

root = os.path.dirname(__file__)

"""returns = load_chunks(os.path.join(root, 'data'), 'portfolio_returns_gamma')
returns.set_index(["gamma", "date"], inplace=True)

frontiers = load_chunks(os.path.join(root, 'data'), 'efficient_frontiers_gamma')
frontiers.set_index(["gamma", "date", "portfolio"], inplace=True)

rates = load_chunks(os.path.join(root, 'data'), 'rf_rate')

print(returns)

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


list_data_equity_path = os.path.join(root, 'data', 'list_equity')

list_data_equity_amer = pd.read_csv(os.path.join(list_data_equity_path, "equity_amer.csv"))
list_data_equity_amer=list_data_equity_amer["ISIN"]

list_data_equity_em = pd.read_csv(os.path.join(list_data_equity_path, "equity_em.csv"))
list_data_equity_em = list_data_equity_em["ISIN"]

list_data_equity_eur = pd.read_csv(os.path.join(list_data_equity_path, "equity_eur.csv"))
list_data_equity_eur = list_data_equity_eur["ISIN"]

list_data_equity_pac = pd.read_csv(os.path.join(list_data_equity_path, "equity_pac.csv"))
list_data_equity_pac = list_data_equity_pac["ISIN"]

master_data_full = load_data(get_path('DS_RI_T_USD_M.xlsx'), cols=lambda x: x != 'NAME', transpose=True)


returns = load_chunks(os.path.join(root, 'data'), 'portfolio_returns_gamma')
returns.set_index(["gamma", "date"], inplace=True)

frontiers = load_chunks(os.path.join(root, 'data'), 'efficient_frontiers_gamma')
frontiers.set_index(["gamma", "date", "portfolio"], inplace=True)

# Sort the index to prevent PerformanceWarning
#frontiers.sort_index(inplace=True)
erc_portfolio_columns = [
        "equity_amer", "equity_em", "equity_eur", "equity_pac",
        "metals", "commodities", "crypto", "volatilities"
    ]
# Filter data for gamma = -0.5
gamma_value = -0.5
if gamma_value in frontiers.index.get_level_values('gamma'):
    # Access volatilities data for the specific gamma value
    Wheights = frontiers.loc[(gamma_value, slice(None), 'erc'), :]
    print(f"Volatilities for gamma = {gamma_value}:")
    print(Wheights[erc_portfolio_columns])
else:
    print(f"No data found for gamma = {gamma_value}")



# Ensure frontiers DataFrame is sorted
frontiers.sort_index(inplace=True)


"""# Filter data for gamma = -0.5
gamma_value = -0.5

if gamma_value in frontiers.index.get_level_values('gamma'):
    # Access data for the specific gamma value and portfolio type
    Wheights = frontiers.loc[(gamma_value, slice(None), 'erc'), :]

    # Format the ISIN list to match the column names
    # Assuming column names in Wheights match the format of "Bitcoin", "Ethereum", etc.
    formatted_columns = [col for col in list_data_equity_amer if col in Wheights.columns]

    if not formatted_columns:
        raise ValueError("No matching columns found between Wheights and list_data_equity_amer.")

    # Filter Wheights with the formatted columns
    filtered_weights = Wheights[formatted_columns]

    print(f"Filtered weights for gamma = {gamma_value}:")
    print(filtered_weights)
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





"""
if choice == "Sub-Portfolio":

    # Portfolio types and clean names
    list_type_portfolio = ['equity_amer', 'equity_em', 'equity_eur', 'equity_pac',
                           'metals', 'commodities', 'crypto', 'volatilities', "erc"]
    list_clean_name = ['Metals', 'Commodities', 'Crypto', 'Volatilities',
                       'North American Equities', 'Emerging Markets Equities',
                       'European Equities', 'Asia-Pacific Equities', "ERC"]

    st.title("Sub-Portfolio")
    selection = st.selectbox("Choose portfolio class", list_clean_name, index=0)
    st.info("We have to optimize each class of portfolio before using Markowitz in our global portfolio")

    # Mapping selection to portfolio name
    selection_to_portfolio_name = {
        "Metals": "metals",
        "Commodities": "commodities",
        "Crypto": "crypto",
        "Volatilities": "volatilities",
        "North American Equities": "equity_amer",
        "Emerging Markets Equities": "equity_em",
        "European Equities": "equity_eur",
        "Asia-Pacific Equities": "equity_pac",
        "ERC": "erc"
    }

    gamma_value = st.session_state.get('gamma_value', None)
    if gamma_value is None:
        st.warning("Please set your gamma in the 'Risk Profiling' section.")
        st.stop()

    portfolio_name = selection_to_portfolio_name[selection]

    # Define sub-portfolio list
    sub_portfolio_list = []
    if selection in ["Metals", "Commodities", "Crypto", "Volatilities"]:
        sub_portfolio_list = globals()[f"list_{portfolio_name}"]
    elif selection in ["North American Equities", "Emerging Markets Equities",
                       "European Equities", "Asia-Pacific Equities"]:
        sub_portfolio_list = globals()[f"list_data_{portfolio_name}"]

    # Load weights and returns data
    try:
        weights_data = load_weights_data(sub_portfolio_list, portfolio_name, gamma_value)
        if selection in ["Metals", "Commodities", "Crypto", "Volatilities"]:
            returns_data = master_df[sub_portfolio_list].pct_change()
        else:
            returns_data = master_data_full[sub_portfolio_list].pct_change()

        # Ensure proper date formatting
        weights_data.index = pd.to_datetime(weights_data.index.get_level_values('date'))
        returns_data.index = pd.to_datetime(returns_data.index)

        # Filter data starting from 1 January 2006
        weights_data = weights_data[weights_data.index >= pd.Timestamp('2006-01-01')]
        returns_data = returns_data[returns_data.index >= pd.Timestamp('2006-01-01')]

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Align data to the first day of the month
    rebalancing_dates = weights_data.index.sort_values()
    weights_monthly = weights_data.resample('MS').ffill()
    returns_data_bom = returns_data.resample('MS').first()

    # Initialize weights if no weights available for the first date
    if not weights_monthly.empty and not weights_monthly.index[0] == returns_data.index[0]:
        first_weights = weights_monthly.iloc[0]
        first_date = returns_data.index[0]
        weights_monthly.loc[first_date] = first_weights
        weights_monthly = weights_monthly.sort_index()

    # Combine returns and weights into a single DataFrame
    dynamic_weights = pd.DataFrame(index=returns_data.index, columns=sub_portfolio_list)
    current_weights = weights_monthly.iloc[0] if not weights_monthly.empty else pd.Series(1 / len(sub_portfolio_list), index=sub_portfolio_list)

    # Update weights dynamically using returns
    for date in returns_data.index:
        if date in weights_monthly.index:  # Rebalance on rebalancing dates
            current_weights = weights_monthly.loc[date]
        portfolio_value = (current_weights * (1 + returns_data.loc[date].fillna(0))).sum()
        current_weights = (current_weights * (1 + returns_data.loc[date].fillna(0))) / portfolio_value
        dynamic_weights.loc[date] = current_weights

    # Fill missing weights forward
    dynamic_weights.fillna(method='ffill', inplace=True)

    # Use select_slider for available dates
    available_dates = dynamic_weights.index[dynamic_weights.index >= pd.Timestamp('2006-01-01')].to_pydatetime()
    if not available_dates.size:
        st.error("No data available from January 2006 onwards.")
        st.stop()

    selected_date = st.select_slider(
        "Select Date",
        options=available_dates,
        value=available_dates[0]
    )
    selected_date = pd.Timestamp(selected_date)

    # Filter data for the selected date
    closest_date = dynamic_weights.index.asof(selected_date)
    if pd.isna(closest_date):
        st.error(f"No data available for the selected date ({selected_date.strftime('%Y-%m-%d')}).")
        st.stop()

    # Retrieve weight allocation for the selected date
    latest_data = dynamic_weights.loc[closest_date]

    # Check if weights are valid (non-empty and normalized)
    if latest_data.empty or latest_data.sum() == 0:
        st.warning("No valid weights available for the selected date.")
        st.stop()

    # Create the pie chart
    fig = px.pie(
        values=latest_data.values,
        names=latest_data.index,
        title=f"Weight Allocation on {closest_date.strftime('%Y-%m-%d')}"
    )
    st.plotly_chart(fig)

    # Display weight allocation over time
    st.subheader("Weight Allocation Over Time")
    st.bar_chart(dynamic_weights)
"""