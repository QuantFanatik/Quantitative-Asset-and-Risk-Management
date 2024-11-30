import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os

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


root = os.path.dirname(__file__)

@st.cache_data
def load_portfolio_returns():
    data = load_chunks(os.path.join(root, 'data'), 'portfolio_returns_gamma', parse_dates=["date"])
    data.set_index(["gamma", "date"], inplace=True)
    return data

@st.cache_data
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
list_data_equity_amer=list_data_equity_amer["ISIN"]

list_data_equity_em = pd.read_csv(os.path.join(list_data_equity_path, "equity_em.csv"))
list_data_equity_em = list_data_equity_em["ISIN"]

list_data_equity_eur = pd.read_csv(os.path.join(list_data_equity_path, "equity_eur.csv"))
list_data_equity_eur = list_data_equity_eur["ISIN"]

list_data_equity_pac = pd.read_csv(os.path.join(list_data_equity_path, "equity_pac.csv"))
list_data_equity_pac = list_data_equity_pac["ISIN"]

master_data_full = load_data(get_path('DS_RI_T_USD_M.xlsx'), cols=lambda x: x != 'NAME', transpose=True)

df_equity_amer = master_data_full[list_data_equity_amer]
df_equity_em = master_data_full[list_data_equity_em]
df_equity_eur = master_data_full[list_data_equity_eur]
df_equity_pac = master_data_full[list_data_equity_pac]

df_equity_amer = df_equity_amer.pct_change()
df_equity_em = df_equity_em.pct_change()
df_equity_eur = df_equity_eur.pct_change()
df_equity_pac = df_equity_pac.pct_change()



# Efficient Frontier Functions
# ---------------------------------------------------------------------------------------
@st.cache_data
def get_gamma_values(data):
    return data.index.get_level_values('gamma').unique()

def get_nearest_gamma(gamma_value, gamma_values):
    gamma_values_array = np.array(gamma_values)
    idx = (np.abs(gamma_values_array - gamma_value)).argmin()
    return gamma_values_array[idx]

def plot_efficient_frontier(data, selected_portfolio, selected_date, gamma_value, risk_free_rate_data):
    # Filter data for the selected gamma, portfolio, and date
    try:
        data_to_plot = data.xs((slice(None), selected_date, selected_portfolio), level=('gamma', 'date', 'portfolio'))
    except KeyError:
        return None  # Return None if data is not available for the selection

    # Extract expected variance and return
    x = data_to_plot['expected_variance'].values
    y = data_to_plot['expected_return'].values
    gamma_values = data_to_plot.index.get_level_values('gamma').values

    # Compute standard deviation from variance
    standard_deviation = np.sqrt(x)

    # Use standard deviation for plotting
    x_plot = standard_deviation
    y_plot = y

    # Get the risk-free rate for the selected date
    if selected_date in risk_free_rate_data.index:
        risk_free_rate = risk_free_rate_data.loc[selected_date, 'rf_rate']
    else:
        # Use the most recent risk-free rate before the selected date
        previous_dates = risk_free_rate_data.index[risk_free_rate_data.index <= selected_date]
        if not previous_dates.empty:
            risk_free_rate = risk_free_rate_data.loc[previous_dates[-1], 'rf_rate']
        else:
            # Default to zero if no rate is available
            risk_free_rate = 0

    # Convert risk-free rate to decimal
    risk_free_rate = risk_free_rate

    # Compute Sharpe ratios
    sharpe_ratios = (y - risk_free_rate) / standard_deviation

    # Find the maximum Sharpe ratio point
    max_sharpe_idx = np.argmax(sharpe_ratios)
    x_sharpe = x[max_sharpe_idx]
    y_sharpe = y[max_sharpe_idx]
    gamma_sharpe = gamma_values[max_sharpe_idx]
    x_sharpe_std = np.sqrt(x_sharpe)

    # Sort the data for plotting
    sorted_indices = np.argsort(gamma_values)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    gamma_sorted = gamma_values[sorted_indices]
    x_plot_sorted = np.sqrt(x_sorted)

    # Plot the efficient frontier
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_plot_sorted,
        y=y_sorted,
        mode='lines+markers',
        name=f"{selected_portfolio.capitalize()} ({selected_date.date()})",
        hovertemplate="Std Dev: %{x:.4f}<br>Expected Return: %{y:.4f}<br>Gamma: %{customdata}",
        customdata=gamma_sorted
    ))

    # Plot the red dot at the user's gamma
    if gamma_value is not None:
        closest_gamma_idx = (np.abs(gamma_sorted - gamma_value)).argmin()
        x_gamma = x_plot_sorted[closest_gamma_idx]
        y_gamma = y_sorted[closest_gamma_idx]
        fig.add_trace(go.Scatter(
            x=[x_gamma],
            y=[y_gamma],
            mode='markers',
            marker=dict(color='red', size=10),
            name=f"Your Portfolio (Gamma={gamma_value:.4f})",
            hovertemplate="Std Dev: %{x:.4f}<br>Expected Return: %{y:.4f}<br>Gamma: %{customdata}",
            customdata=[gamma_sorted[closest_gamma_idx]]
        ))
    else:
        st.warning("Please set your Gamma in the 'Risk Profiling' section.")

    # Plot the green dot at the maximum Sharpe ratio point
    fig.add_trace(go.Scatter(
        x=[x_sharpe_std],
        y=[y_sharpe],
        mode='markers',
        marker=dict(color='green', size=10),
        name="Max Sharpe Ratio Portfolio",
        hovertemplate="Std Dev: %{x:.4f}<br>Expected Return: %{y:.4f}<br>Gamma: %{customdata}",
        customdata=[gamma_sharpe]
    ))

    # Plot the white dot at the risk-free rate
    fig.add_trace(go.Scatter(
        x=[0],
        y=[risk_free_rate],
        mode='markers',
        marker=dict(color='white', size=8),
        name="Risk-Free Rate",
        hovertemplate="Std Dev: 0<br>Expected Return: %{y:.4f}<br>",
    ))

    # Plot the Capital Market Line (CML)
    # Calculate the slope of the CML (Sharpe ratio of the tangency portfolio)
    cml_slope = (y_sharpe - risk_free_rate) / x_sharpe_std

    # Generate points for the CML
    cml_x = np.linspace(0, x_plot_sorted.max(), 100)
    cml_y = risk_free_rate + cml_slope * cml_x

    # Plot the CML
    fig.add_trace(go.Scatter(
        x=cml_x,
        y=cml_y,
        mode='lines',
        name="Capital Market Line (CML)",
        line=dict(color='blue', dash='dash'),
        hoverinfo='skip'
    ))

    # Update the layout
    fig.update_layout(
        title=f"Efficient Frontier and CML for {selected_portfolio.capitalize()} on {selected_date.date()}",
        xaxis_title="Standard Deviation (Risk)",
        yaxis_title="Expected Return",
        width=800,
        height=600
    )
    return fig

# ***********************************************************************************************************
# Principal bar
# ***********************************************************************************************************
with st.sidebar:
    st.title("Portfolio Optimization")
    choice = st.radio("Steps", ["Risk Profiling", "Data Exploration", "Sub-Portfolio", "Optimal Portfolio", "Performance"])
    st.info("This tool uses equal risk contributions method to select optimal weights for each "
            "type of asset classes. Then, we use Markowitz optimization to choose an optimal portfolio.")

# ***********************************************************************************************************
# Risk Profiling
# ***********************************************************************************************************
if choice == "Risk Profiling":

    st.title("Risk Profiling")
    gamma_known = st.radio("Do you know your Gamma?", ("Yes", "No"))

    if gamma_known == "Yes":
        gamma_value = st.number_input("Please enter your Gamma value", min_value=-1.0, max_value=None, value=1.5)
        st.write(f"You have entered Gamma = {gamma_value}")

    else:
        st.write("We will ask you some questions to help determine your Gamma.")
        # Questions to define the Gamma
        q1 = st.slider("On a scale from 1 (Very low risk tolerance) to 5 (Very high risk tolerance), how would you rate your risk tolerance?", 1, 5, 3)
        q2 = st.selectbox("What is your investment horizon?", ["Short-term (less than 3 years)", "Medium-term (3-7 years)", "Long-term (more than 7 years)"])
        q3 = st.selectbox("What is your primary investment goal?", ["Capital preservation", "Income generation", "Growth"])
        q4 = st.selectbox("How would you react if your investment portfolio declined by 20% over a single month?", ["Completely panicked", "Stressed but not panicked", "It happens"])
        q5 = st.selectbox("How stable is your current income stream?", ["Very unstable", "Somewhat unstable", "Stable", "Very stable"])

        # Assign numerical values to the answers
        risk_tolerance_score = q1

        if q2 == "Short-term (less than 3 years)":
            horizon_score = 5
        elif q2 == "Medium-term (3-7 years)":
            horizon_score = 2
        else:
            horizon_score = 1

        if q3 == "Capital preservation":
            goal_score = 1
        elif q3 == "Income generation":
            goal_score = 3
        else:
            goal_score = 5

        if q4 == "Completely panicked":
            goal_score_1 = 1
        elif q4 == "Stressed but not panicked":
            goal_score_1 = 3
        else:
            goal_score_1 = 5

        if q5 == "Very unstable":
            goal_score_2 = 1
        elif q5 == "Somewhat unstable":
            goal_score_2 = 3
        elif q5 == "Stable":
            goal_score_2 = 5
        else:
            goal_score_2 = 8

        # Calculate Gamma (this is a simplified example; adjust as needed)
        gamma_score = risk_tolerance_score + horizon_score + goal_score + goal_score_1 + goal_score_2
        gamma_value = (23 / gamma_score) * 0.5  # Higher score implies lower gamma

        st.write(f"Based on your answers, your estimated Gamma is **{gamma_value:.4f}**")

    # Adjust gamma_value to the nearest gamma in the dataset
    # Load the data to get gamma values
    frontier_data = load_efficient_frontier_data()
    gamma_values_in_data = get_gamma_values(frontier_data)
    gamma_value = get_nearest_gamma(gamma_value, gamma_values_in_data)
    st.write(f"Your Gamma value has been adjusted to the nearest available gamma: **{gamma_value:.4f}**")
    st.session_state['gamma_value'] = gamma_value

    st.success("Your Gamma value has been set. You can proceed to other sections.")

# ***********************************************************************************************************
# Data Exploration
# ***********************************************************************************************************
if choice == "Data Exploration":

    st.title("Data Exploration")
    selection = st.selectbox("Choose portfolio class", list_clean_name, index=0)

    # Map the selection to the portfolio name used in the data
    portfolio_mapping = {
        'Metals': 'metals',
        'Commodities': 'commodities',
        'Crypto': 'crypto',
        'Volatilities': 'volatilities',
        'North American Equities': 'equity_amer',
        'Emerging Markets Equities': 'equity_em',
        'European Equities': 'equity_eur',
        'Asia-Pacific Equities': 'equity_pac',
    }
    selected_portfolio = portfolio_mapping.get(selection)

    if selection in ["Commodities", "Metals", "Crypto", "Volatilities"]:

        if selection == "Commodities":
            data_use = df_commodities
            correl_matrix = corr_matrix.loc[list_commodities, list_commodities]

        elif selection == "Metals":
            data_use = df_metals
            correl_matrix = corr_matrix.loc[list_metals, list_metals]

        elif selection == "Crypto":
            data_use = df_crypto
            correl_matrix = corr_matrix.loc[list_crypto, list_crypto]

        elif selection == "Volatilities":
            data_use = df_volatilities
            correl_matrix = corr_matrix.loc[list_volatilities, list_volatilities]

        # Display expected returns and volatilities
        st.subheader("Expected Annualized Returns and Volatilities", divider="gray")
        st.write("Expected Annualized Returns")
        st.bar_chart(data_use.mean() * 252)
        st.write("Expected Annualized Volatilities")
        st.bar_chart(data_use.std() * np.sqrt(252))

        # Heatmap visualization
        st.subheader("Correlation Heatmap", divider="gray")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            correl_matrix,
            annot=True,
            cmap="coolwarm",
            annot_kws={"color": "white"},
            xticklabels=correl_matrix.columns,
            yticklabels=correl_matrix.index,
            ax=ax
        )
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        ax.tick_params(axis='x', colors='white', labelrotation=45)
        ax.tick_params(axis='y', colors='white', labelrotation=0)
        st.pyplot(fig)

        # Efficient Frontier for the selected portfolio
        st.subheader("Efficient Frontier")
        frontier_data = load_efficient_frontier_data()
        rf_rate_data = load_rates_data()

        if frontier_data.empty or rf_rate_data.empty:
            st.error("Efficient frontier data or risk-free rate data is unavailable.")
        else:
            try:
                gamma_value = st.session_state.get('gamma_value', None)
                if gamma_value is None:
                    st.warning("Please set your gamma in the 'Risk Profiling' section.")
                else:
                    # Get available dates for the selected portfolio
                    available_data = frontier_data.xs(selected_portfolio, level='portfolio')
                    available_dates = available_data.index.get_level_values('date').unique().sort_values()
                    selected_date = st.select_slider(
                        "Select Date",
                        options=available_dates,
                        value=available_dates[0]
                    )
                    fig_ef = plot_efficient_frontier(
                        frontier_data, selected_portfolio, selected_date, gamma_value, rf_rate_data
                    )
                    if fig_ef is not None:
                        st.plotly_chart(fig_ef)
                    else:
                        st.error(f"No data available for {selection} on {selected_date}")
            except KeyError:
                st.error(f"No data available for {selection}")
                st.stop()

    elif selection in ['North American Equities', 'Emerging Markets Equities', 'European Equities',
                       'Asia-Pacific Equities']:

        st.info("Data exploration is different for equities due to the large number of securities.")

        # Load portfolio returns
        portfolio_returns = load_portfolio_returns()
        gamma_value = st.session_state.get('gamma_value', None)
        if gamma_value is None:
            st.warning("Please set your gamma in the 'Risk Profiling' section.")
            st.stop()
        else:
            # Get returns for the selected gamma
            try:
                returns_gamma = portfolio_returns.xs(gamma_value, level='gamma')
            except KeyError:
                st.error(f"No data available for gamma value {gamma_value}")
                st.stop()

        if selection == "North American Equities":
            mean = str(round(float(returns_gamma["equity_amer"].mean() * 12 * 100), 2))
            vol = str(round(float(returns_gamma["equity_amer"].std() * np.sqrt(12) * 100), 2))
            nb_eq = int(list_data_equity_amer.shape[0])
            list_data_equity_amer.rename(columns={"equity_amer": "ISIN"}, inplace=True)
            list_isin = list_data_equity_amer

        elif selection == "Emerging Markets Equities":
            mean = str(round(float(returns_gamma["equity_em"].mean() * 12 * 100), 2))
            vol = str(round(float(returns_gamma["equity_em"].std() * np.sqrt(12) * 100), 2))
            nb_eq = int(list_data_equity_em.shape[0])
            list_isin = list_data_equity_em

        elif selection == "European Equities":
            mean = str(round(float(returns_gamma["equity_eur"].mean() * 12 * 100), 2))
            vol = str(round(float(returns_gamma["equity_eur"].std() * np.sqrt(12) * 100), 2))
            nb_eq = int(list_data_equity_eur.shape[0])
            list_isin = list_data_equity_eur

        elif selection == "Asia-Pacific Equities":
            mean = str(round(float(returns_gamma["equity_pac"].mean() * 12 * 100), 2))
            vol = str(round(float(returns_gamma["equity_pac"].std() * np.sqrt(12) * 100), 2))
            nb_eq = int(list_data_equity_pac.shape[0])
            list_isin = list_data_equity_pac

        # Display expected return, volatility, and equity details
        st.subheader("Expected Annualized Return and Volatility", divider="gray")
        st.markdown(f"**Expected Annualized Return**: {mean}%")
        st.markdown(f"**Expected Annualized Volatility**: {vol}%")
        st.write("")
        st.subheader("Additional Information", divider="gray")
        st.markdown(f"**Number of Equities**: {nb_eq}")
        st.write("")
        st.markdown("**Equities Composition**:")
        st.write(list_isin)

        # Efficient Frontier for equity portfolios
        st.subheader("Efficient Frontier")
        frontier_data = load_efficient_frontier_data()
        rf_rate_data = load_rates_data()

        if frontier_data.empty or rf_rate_data.empty:
            st.error("Efficient frontier data or risk-free rate data is unavailable.")
        else:
            try:
                gamma_value = st.session_state.get('gamma_value', None)
                if gamma_value is None:
                    st.warning("Please set your gamma in the 'Risk Profiling' section.")
                else:
                    # Get available dates for the selected portfolio
                    available_data = frontier_data.xs(selected_portfolio, level='portfolio')
                    available_dates = available_data.index.get_level_values('date').unique().sort_values()
                    selected_date = st.select_slider(
                        "Select Date",
                        options=available_dates,
                        value=available_dates[0]
                    )
                    fig_ef = plot_efficient_frontier(
                        frontier_data, selected_portfolio, selected_date, gamma_value, rf_rate_data
                    )
                    if fig_ef is not None:
                        st.plotly_chart(fig_ef)
                    else:
                        st.error(f"No data available for {selection} on {selected_date}")
            except KeyError:
                st.error(f"No data available for {selection}")
                st.stop()

# ***********************************************************************************************************
# Sub-Portfolio
# ***********************************************************************************************************
if choice == "Sub-Portfolio":

    st.title("Sub-Portfolio")
    selection = st.selectbox("Choose portfolio class", list_clean_name, index=0)
    st.info("We have to optimize each class of portfolio before using Markowitz in our global portfolio")

    # Map the selection to the portfolio name used in the DataFrame
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

    # Load weights data for the selected portfolio
    if selection in ["Metals", "Commodities", "Crypto", "Volatilities", "North American Equities", "Emerging Markets Equities", "European Equities", "Asia-Pacific Equities", "ERC"]:
        if selection == "Metals":
            sub_portfolio_list = list_metals
        elif selection == "Commodities":
            sub_portfolio_list = list_commodities
        elif selection == "Crypto":
            sub_portfolio_list = list_crypto
        elif selection == "Volatilities":
            sub_portfolio_list = list_volatilities
        elif selection == "North American Equities":
            sub_portfolio_list = list_data_equity_amer
        elif selection == "Emerging Markets Equities":
            sub_portfolio_list = list_data_equity_em
        elif selection == "European Equities":
            sub_portfolio_list = list_data_equity_eur
        elif selection == "Asia-Pacific Equities":
            sub_portfolio_list = list_data_equity_pac
        elif selection == "ERC" :
            sub_portfolio_list = list_ERC
    else:
        st.error("Sub-Portfolio Analysis for equities is not implemented in this section.")
        st.stop()

    weights_data = load_weights_data(sub_portfolio_list, portfolio_name, gamma_value)
    if selection in ["Metals", "Commodities", "Crypto", "Volatilities"]:
        returns_data = master_df[sub_portfolio_list].pct_change()
    elif selection in ["North American Equities", "Emerging Markets Equities", "European Equities", "Asia-Pacific Equities"]:
        returns_data = master_data_full[sub_portfolio_list].pct_change()

    # Ensure the index is properly set to dates
    weights_data.index = pd.to_datetime(weights_data.index.get_level_values('date'))
    returns_data.index = pd.to_datetime(returns_data.index)

    # Align data to the first day of the month
    rebalancing_dates = weights_data.index.sort_values()
    weights_monthly = weights_data.loc[rebalancing_dates].resample('MS').ffill()
    returns_data_bom = returns_data.resample('MS').first()

    # Combine returns and weights into a single DataFrame
    dynamic_weights = pd.DataFrame(index=returns_data_bom.index, columns=sub_portfolio_list)
    for date in returns_data_bom.index:
        if date in weights_monthly.index:
            dynamic_weights.loc[date] = weights_monthly.loc[date]
    dynamic_weights.fillna(method='ffill', inplace=True)

    # Use select_slider for first day of monthly dates
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


# ***********************************************************************************************************
# Portfolio Optimization
# ***********************************************************************************************************
if choice == "Optimal Portfolio":

    st.title("Portfolio Optimization")
    st.info("Optimize the global portfolio using the efficient frontier data.")

    # Map the ERC portfolio to relevant sub-portfolio columns
    erc_portfolio_columns = [
        "equity_amer", "equity_em", "equity_eur", "equity_pac",
        "metals", "commodities", "crypto", "volatilities"
    ]

    gamma_value = st.session_state.get('gamma_value', None)
    if gamma_value is None:
        st.warning("Please set your Gamma in the 'Risk Profiling' section.")
        st.stop()

    # Load the efficient frontier data
    frontier_data = load_efficient_frontier_data()

    # Filter data for the selected gamma and ERC portfolio
    try:
        portfolio_data = frontier_data.loc[(gamma_value, slice(None), "erc"), erc_portfolio_columns]
    except KeyError:
        st.error(f"No data available for gamma {gamma_value} and ERC portfolio.")
        st.stop()

    # Ensure the index is properly set to datetime and align to the first day of the month
    portfolio_data.index = pd.to_datetime(portfolio_data.index.get_level_values("date"))
    portfolio_monthly = portfolio_data.resample('MS').first()

    # Use select_slider for first day of monthly dates
    available_dates = portfolio_monthly.index[portfolio_monthly.index >= pd.Timestamp('2006-01-01')].to_pydatetime()
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
    closest_date = portfolio_monthly.index.asof(selected_date)
    if pd.isna(closest_date):
        st.error(f"No data available for the selected date ({selected_date.strftime('%Y-%m-%d')}).")
        st.stop()

    latest_data = portfolio_monthly.loc[closest_date]

    # Create dynamic weights DataFrame
    dynamic_weights = pd.DataFrame(index=portfolio_monthly.index, columns=erc_portfolio_columns)
    dynamic_weights.loc[closest_date] = latest_data

    # Fill forward any missing weights
    dynamic_weights.fillna(method='ffill', inplace=True)

    # Extract weights for the selected date
    latest_weights = latest_data[erc_portfolio_columns]
    latest_weights = latest_weights[latest_weights > 0]  # Keep only positive weights

    # Create the pie chart for weight allocation
    fig = px.pie(
        values=latest_weights.values,
        names=latest_weights.index,
        title=f"Weight Allocation on {closest_date.strftime('%Y-%m-%d')}"
    )
    st.plotly_chart(fig)

    # Display weight allocation over time
    st.subheader("Weight Allocation Over Time")
    st.area_chart(dynamic_weights)


# ***********************************************************************************************************
# Performance
# ***********************************************************************************************************
if choice == "Performance":
    st.title("Performance")
    st.subheader("Cumulative Returns and Drawdown")

    list_portfolios = [
        'equity_amer', 'equity_em', 'equity_eur', 'equity_pac',
        'metals', 'commodities', 'volatilities', 'crypto', 'erc'
    ]
    selected_portfolios = []

    # Portfolio Selection
    st.write("Select portfolios to display:")
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

        # Plot cumulative returns
        st.write("### Cumulative Returns")
        st.line_chart(cumulative_returns_selected)

        # Calculate drawdown
        st.write("### Drawdown")
        cumulative_returns_max = cumulative_returns_selected.cummax()
        drawdowns = (cumulative_returns_selected - cumulative_returns_max) / cumulative_returns_max
        st.line_chart(drawdowns)

        # Correlation Heatmap
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
        st.write("### Summary Statistics")
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

        st.info("Hover over the rows to highlight them. Metrics are color-coded for easier comparison.")
    else:
        st.warning("Please select at least one portfolio to view its performance.")
