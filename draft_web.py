import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os
import gc

pd.set_option('future.no_silent_downcasting', True)
root = os.path.dirname(__file__)

# Data Loading Functions
# @st.cache_data
def optimize_floats(df, robust=False):
    if robust:
        return df.astype('float16')
    if isinstance(df, pd.Series):
        if df.dtype in ['float64', 'float32']:
            if df.max() <= 65000 and df.min() >= -65000:
                return df.astype('float16')
            else:
                # print(f"Warning: {df.name} has values outside the range of float16.")
                return df.astype('float32')  # Fallback to float32 if out of range
        else:
            return df 
    elif isinstance(df, pd.DataFrame):
        for col in df.select_dtypes(include=['float']).columns:
            if df[col].max() <= 65000 and df[col].min() >= -65000:
                df[col] = df[col].astype('float16')
            else:
                # print(f"Warning: {col} has values outside the range of float16.")
                df[col] = df[col].astype('float32')  # Fallback to float32 if out of range
        return df
    else:
        raise TypeError("Input must be a pandas DataFrame or Series.")
# ---------------------------------------------------------------------------------------
# @st.cache_data
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

@st.cache_data
def load_portfolio_returns():
    data = load_chunks(os.path.join(root, 'data'), 'portfolio_returns_gamma', parse_dates=["date"])
    data['date'] = pd.to_datetime(data['date'])
    data.set_index(["gamma", "date"], inplace=True)
    return optimize_floats(data)

@st.cache_data
def load_efficient_frontier_data():
    data = load_chunks(os.path.join(root, 'data'), 'efficient_frontiers_gamma', parse_dates=["date"])
    data['date'] = pd.to_datetime(data['date'])
    data.set_index(["gamma", "date", "portfolio"], inplace=True)
    data.sort_index(inplace=True)
    return optimize_floats(data)

@st.cache_data
def load_rates_data():
    data = pd.read_csv(os.path.join(root, 'data/rf_rate.csv'))
    data['date'] = pd.to_datetime(data['date'])
    return data.rename(columns={'RF': 'rf_rate'}).set_index('date')  

@st.cache_data
def load_master_df():
    """Load and process the main dataset."""
    path = os.path.join(root, 'data', 'all_prices.csv')
    df = pd.read_csv(path, skiprows=1, nrows=1)
    columns = ['date', *list(df.columns)[1:]]
    master_df = pd.read_csv(path, skiprows=3, names=columns, parse_dates=['date']).set_index('date')
    del df
    gc.collect()
    return optimize_floats(master_df)


#### Last above
@st.cache_data
def load_correlation_and_returns(master_df):
    """Compute correlation matrix and returns."""
    corr_matrix = optimize_floats(master_df.corr())
    master_returns = master_df.pct_change(fill_method=None).replace([np.inf, -np.inf, np.nan], 0)
    master_mean = optimize_floats(master_returns.mean() * 12)
    master_std = optimize_floats(master_returns.std() * np.sqrt(12))
    master_returns = optimize_floats(master_returns)
    return corr_matrix, master_returns, master_mean, master_std


# Exception
@st.cache_data
def load_equity_lists():
    """Load equity lists for different regions."""
    list_data_equity_path = os.path.join(root, 'data', 'list_equity')
    equity_amer = pd.read_csv(os.path.join(list_data_equity_path, "equity_amer.csv"))["ISIN"]
    equity_em = pd.read_csv(os.path.join(list_data_equity_path, "equity_em.csv"))["ISIN"]
    equity_eur = pd.read_csv(os.path.join(list_data_equity_path, "equity_eur.csv"))["ISIN"]
    equity_pac = pd.read_csv(os.path.join(list_data_equity_path, "equity_pac.csv"))["ISIN"]
    return equity_amer, equity_em, equity_eur, equity_pac

@st.cache_data
def process_portfolio_returns(portfolio_returns):
    """Process portfolio return data."""
    portfolio_mean = optimize_floats(portfolio_returns.groupby('gamma').mean() * 12 * 100, robust=True)
    portfolio_std = optimize_floats(portfolio_returns.groupby('gamma').std() * np.sqrt(12) * 100, robust=True)
    return portfolio_mean, portfolio_std

@st.cache_data
def extract_available_dates(frontier_data):
    """Extract unique sorted available dates from the frontier data."""
    available_dates = frontier_data.index.get_level_values('date').unique().sort_values()
    return available_dates

@st.cache_data
def process_portfolio_weights(frontier_data):
    """Process portfolio weights data."""
    all_weights_data = frontier_data.drop(columns=['expected_return', 'expected_variance'])
    all_weights_data = all_weights_data.where(all_weights_data > 0, 0)
    all_weights_data = all_weights_data.div(all_weights_data.sum(axis=1), axis=0)
    all_weights_data = optimize_floats(all_weights_data, robust=True)
    return all_weights_data

# st.session_state['gamma_value'] = 0.5  # Default value

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

    # # Use standard deviation for plotting
    # x_plot = standard_deviation
    # y_plot = y

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
        title=f"Efficient Frontier for {selected_portfolio.capitalize()} on {selected_date.date()}",
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
    choice = st.radio("Steps", ["Introduction", "Risk Profiling", "Data Exploration", "Sub-Portfolio", "Final Portfolio", "Performance"])

gc.collect()

equity_amer, equity_em, equity_eur, equity_pac = load_equity_lists()
list_type_portfolio = ['equity_amer', 'equity_em', 'equity_eur', 'equity_pac', 'metals', 'commodities', 'crypto', 'volatilities']

list_clean_name = ['Metals', 'Commodities', 'Crypto', 'Volatilities', 
                   'North American Equities', 'Emerging Markets Equities', 
                   'European Equities', 'Asia-Pacific Equities']

list_commodities = ["Lean_Hogs", "Crude_Oil", "Live_Cattle", "Soybeans", "Wheat", "Corn", "Natural_Gas"]

list_crypto = ["Bitcoin", "Ethereum"]

list_metals = ["Gold", "Platinum", "Palladium", "Silver", "Copper"]

list_volatilities = ["Russell_2000_RVX", "VVIX_VIX_of_VIX", "MOVE_bond_market_volatility", 
                     "VXO-S&P_100_volatility", "Nasdaq_VXN", "VIX"]

list_ERC = list_type_portfolio

ticker_mapping = {
        'Metals': list_metals,
        'Commodities': list_commodities,
        'Crypto': list_crypto,
        'Volatilities': list_volatilities,
        'North American Equities': equity_amer,
        'Emerging Markets Equities': equity_em,
        'European Equities': equity_eur,
        'Asia-Pacific Equities': equity_pac,
        'ERC': list_ERC
    }

portfolio_mapping = {
        'Metals': 'metals',
        'Commodities': 'commodities',
        'Crypto': 'crypto',
        'Volatilities': 'volatilities',
        'North American Equities': 'equity_amer',
        'Emerging Markets Equities': 'equity_em',
        'European Equities': 'equity_eur',
        'Asia-Pacific Equities': 'equity_pac',
        'ERC': 'erc'
    }

frontier_data = load_efficient_frontier_data()
gamma_array = frontier_data.index.get_level_values('gamma').unique().values
def get_nearest_gamma(gamma_value):
    return min(gamma_array, key=lambda x: abs(x - gamma_value))
del frontier_data


# ***********************************************************************************************************
# Introduction
# ***********************************************************************************************************
if choice == "Introduction":

    st.title("Portfolio Optimization Web Application")
    st.subheader("An Intelligent Tool for Building Optimal Investment Portfolios")

    st.markdown("""
    Welcome to our portfolio optimization web application! This platform is designed to help investors construct efficient portfolios tailored to their risk preferences.
    """)

    st.markdown("### Steps Involved in the Project")
    st.markdown("""
    Our approach consists of the following main steps:
    """)

    # Step 1: Risk Assessment
    st.markdown("#### 1. Risk Assessment")
    st.write("""
    The first step involves determining your **risk aversion parameter (Gamma)**. 
    - You can either directly input your gamma value if you know it.
    - Alternatively, you will be asked a series of questions about your financial goals, investment horizon, and risk tolerance to calculate your gamma value automatically.
    """)

    # Step 2: Asset Class Summary
    st.markdown("#### 2. Asset Class Summary")
    st.write("""
    Once your risk aversion is set, we will provide a **quick overview of the key asset classes** available for investment:
    - **Metals**: Precious and industrial metals such as Gold, Silver, and Copper.
    - **Crypto**: Cryptocurrencies like Bitcoin and Ethereum.
    - **Volatilities**: Volatility indices like the VIX  
    - **Commodities**: Agricultural and energy commodities like Crude Oil and Natural Gas.
    - **Equities**: Regional equity markets, including North America, Europe, Emerging Markets, and Asia-Pacific.

    """)

    # Step 3: Portfolio Optimization
    st.markdown("#### 3. Portfolio Optimization")
    st.write("""
    The optimization process consists of two main stages:
    1. **Mean-Variance Optimization (MVO)**: 
        - For each asset class, we perform a **mean-variance optimization** to find an efficient portfolio within that class.
    2. **Equal Risk Contribution (ERC) Portfolio**:
        - We construct a global portfolio where each sub-portfolio (optimized for its respective asset class) contributes equally to the total portfolio risk.
    """)

    # Step 4: Out-of-Sample Performance Analysis
    st.markdown("#### 4. Out-of-Sample Performance Analysis")
    st.write("""
    Finally, we evaluate the performance of the optimized portfolio using **out-of-sample data**. This includes:
    - Analyzing the cumulative returns and drawdowns of the final portfolio.
    - Comparing the performance of the global portfolio to its individual sub-portfolios.
    """)


# ***********************************************************************************************************
# Risk Profiling
# ***********************************************************************************************************
if choice == "Risk Profiling":

    st.title("Risk Profiling")
    gamma_known = st.radio("Do you know your Gamma?", ("Yes", "No"))

    if gamma_known == "Yes":
        gamma_value = st.number_input("Please enter your Gamma value", min_value=-0.5, max_value=None, value=0.5)

    else:
        st.write("We will ask you some questions to help determine your Gamma.")

        # Questions to define the Gamma
        risk_tolerance_score = st.slider(
            "On a scale from 1 (Very low risk tolerance) to 5 (Very high risk tolerance), how would you rate your risk tolerance?",
            0, 3, 5
        )

        investment_horizon = st.selectbox(
            "What is your investment horizon?",
            ["Short-term (less than 3 years)", "Medium-term (3-7 years)", "Long-term (more than 7 years)"]
        )

        primary_goal = st.selectbox(
            "What is your primary investment goal?",
            ["Capital preservation", "Income generation", "Growth"]
        )

        reaction_to_decline = st.selectbox(
            "How would you react if your investment portfolio declined by 20% over a single month?",
            ["Completely panicked", "Stressed but not panicked", "It happens"]
        )

        income_stability = st.selectbox(
            "How stable is your current income stream?",
            ["Very unstable", "Somewhat unstable", "Stable", "Very stable"]
        )

        # Assign scores
        goal_score = {"Capital preservation": 0, "Income generation": 1.5, "Growth": 3}[primary_goal]
        decline_score = {"Completely panicked": 0, "Stressed but not panicked": 1.5, "It happens": 3}[reaction_to_decline]
        income_score = {"Very unstable": 0, "Somewhat unstable": 1, "Stable": 1.5, "Very stable": 3}[income_stability]

        # Calculate Gamma
        gamma_score = risk_tolerance_score + goal_score + decline_score + income_score
        gamma_value = (gamma_score / 15) * 0.5  # Normalized to a range

    # Adjust Gamma value to nearest available Gamma in the dataset
    #TODO: gamma problems
    gamma_value = get_nearest_gamma(gamma_value)
    st.write(f"Based on your answers, your estimated Gamma is **{gamma_value:.4f}**")

    # Save Gamma to session state
    st.session_state['gamma_value'] = gamma_value


# ***********************************************************************************************************
# Data Exploration
# ***********************************************************************************************************
if choice == "Data Exploration":

    master_df = load_master_df()
    corr_matrix, master_returns, master_mean, master_std = load_correlation_and_returns(master_df)
    del master_df

    st.title("Data Exploration")
    selection = st.selectbox("Choose portfolio class", list_clean_name, index=0)

    selected_portfolio = portfolio_mapping.get(selection)

    if selection in ["Commodities", "Metals", "Crypto", "Volatilities"]:

        mean_use = master_mean[ticker_mapping[selection]]
        del master_mean
        std_use = master_std[ticker_mapping[selection]]
        del master_std
        correl_matrix = corr_matrix.loc[ticker_mapping[selection], ticker_mapping[selection]]
        del corr_matrix

        # Display expected returns and volatilities
        st.subheader("Expected Annualized Returns and Volatilities", divider="gray")
        st.write("Expected Annualized Returns")
        st.bar_chart(mean_use)
        st.write("Expected Annualized Volatilities")
        st.bar_chart(std_use)

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
        st.subheader("Efficient Frontier", divider="gray")
        # frontier_data = load_efficient_frontier_data()
        # rf_rate_data = load_rates_data()

        frontier_data = load_efficient_frontier_data()
        rf_rate_data = load_rates_data()
        available_dates = extract_available_dates(frontier_data)
        
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
                    # available_dates = available_data.index.get_level_values('date').unique().sort_values()
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
        # portfolio_returns = load_portfolio_returns()
        portfolio_returns = optimize_floats(load_portfolio_returns(), robust=True)
        portfolio_mean, portfolio_std = process_portfolio_returns(portfolio_returns)

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

        mapping = portfolio_mapping[selection]
        mean = str(round(float(portfolio_mean.loc[gamma_value, mapping]), 2))
        vol = str(round(float(portfolio_std.loc[gamma_value, mapping]), 2))
        del portfolio_mean, portfolio_std
        list_isin = ticker_mapping[selection]
        nb_eq = len(list_isin)

        # Display expected return, volatility, and equity details
        st.subheader("Expected Annualized Return and Volatility", divider="gray")
        st.markdown(f"**Expected Annualized Return**: {mean}%")
        st.markdown(f"**Expected Annualized Volatility**: {vol}%")
        st.write("")
        st.subheader("Additional Information", divider="gray")
        st.markdown(f"**Number of Equities**: {nb_eq}")
        st.markdown("**Equities Composition**:")
        st.write(list_isin)

        # Efficient Frontier for equity portfolios
        st.subheader("Efficient Frontier", divider="gray")
        # frontier_data = load_efficient_frontier_data()
        # rf_rate_data = load_rates_data()

        frontier_data = load_efficient_frontier_data()
        rf_rate_data = load_rates_data()
        available_dates = extract_available_dates(frontier_data)

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
                    # available_dates = available_data.index.get_level_values('date').unique().sort_values()
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

    # Portfolio types and clean names
    list_type_portfolio = ['equity_amer', 'equity_em', 'equity_eur', 'equity_pac',
                           'metals', 'commodities', 'crypto', 'volatilities']
    list_clean_name = ['Metals', 'Commodities', 'Crypto', 'Volatilities',
                       'North American Equities', 'Emerging Markets Equities',
                       'European Equities', 'Asia-Pacific Equities']

    st.title("Sub-Portfolio")
    selection = st.selectbox("Choose portfolio class", list_clean_name, index=0)

    gamma_value = st.session_state.get('gamma_value', None)
    if gamma_value is None:
        st.warning("Please set your gamma in the 'Risk Profiling' section.")
        st.stop()

    portfolio_name = portfolio_mapping[selection]

    if portfolio_name == "crypto":
        limit = '2014-01-01'
    else:
        limit = '2006-01-01'

    sub_portfolio_list = ticker_mapping[selection]

    master_df = load_master_df()
    _, master_returns, _, _ = load_correlation_and_returns(master_df)
    del _
    frontier_data = load_efficient_frontier_data()
    all_weights_data = process_portfolio_weights(frontier_data)
    del frontier_data

    try:
        weights_data = all_weights_data.xs(key=(gamma_value, portfolio_name), level=('gamma', 'portfolio'))[sub_portfolio_list]
        returns_data = master_returns[sub_portfolio_list]
        weights_data = weights_data[weights_data.index >= pd.Timestamp(limit)]
        returns_data = returns_data[returns_data.index >= pd.Timestamp(limit)]

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    rebalancing_dates = weights_data.index.sort_values()
    weights_monthly = weights_data.resample('MS').ffill()

    if not weights_monthly.empty and not weights_monthly.index[0] == returns_data.index[0]:
        first_weights = weights_monthly.iloc[0]
        first_date = returns_data.index[0]
        weights_monthly.loc[first_date] = first_weights
        weights_monthly = weights_monthly.sort_index()

    # Combine returns and weights into a single DataFrame
    dynamic_weights = pd.DataFrame(index=returns_data.index, columns=sub_portfolio_list)
    current_weights = weights_monthly.iloc[0] if not weights_monthly.empty else pd.Series(1 / len(sub_portfolio_list), index=sub_portfolio_list)
    prev_date = 0
    for date in returns_data.index:
        #print(date, date.month)
        if date.month == 1 and date.year <= 2021 and date.year != prev_date:
            new_date = pd.Timestamp(date).replace(day=1)
            current_weights = weights_monthly.loc[new_date]

        portfolio_value = (current_weights * (1 + returns_data.loc[date].fillna(0))).sum()
        current_weights = (current_weights * (1 + returns_data.loc[date].fillna(0))) / portfolio_value

        dynamic_weights.loc[date] = current_weights
        prev_date= date.year

    dynamic_weights.ffill(inplace=True)

    available_dates = dynamic_weights.index[dynamic_weights.index >= pd.Timestamp(limit)].to_pydatetime()
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
    dynamic_weights = dynamic_weights.div(dynamic_weights.sum(axis=1), axis=0).fillna(0)  # Ensure normalization
    st.bar_chart(dynamic_weights)

# ***********************************************************************************************************
# Optimal Portfolio
# ***********************************************************************************************************
if choice == "Final Portfolio":

    st.title("Final Portfolio")

    # Map the ERC portfolio to relevant sub-portfolio columns
    erc_portfolio_columns = [
        "equity_amer", "equity_em", "equity_eur", "equity_pac",
        "metals", "commodities", "crypto", "volatilities"
    ]

    gamma_value = st.session_state.get('gamma_value', None)
    if gamma_value is None:
        st.warning("Please set your Gamma in the 'Risk Profiling' section.")
        st.stop()

    # Define sub-portfolio list for ERC
    sub_portfolio_list = erc_portfolio_columns
    frontier_data = load_efficient_frontier_data()
    all_weights_data = process_portfolio_weights(frontier_data)
    del frontier_data

    try:
        weights_data = all_weights_data.xs(key=(gamma_value, "erc"), level=('gamma', 'portfolio'))[sub_portfolio_list]
        del all_weights_data

        portfolio_returns = optimize_floats(load_portfolio_returns(), robust=True)
        returns_data = portfolio_returns.xs(gamma_value, level='gamma')[sub_portfolio_list]
        del portfolio_returns

        weights_data = weights_data[weights_data.index >= pd.Timestamp('2007-01-01')]
        returns_data = returns_data[returns_data.index >= pd.Timestamp('2007-01-01')]

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Align data to the first day of the month
    rebalancing_dates = weights_data.index.sort_values()
    weights_monthly = weights_data.resample('MS').ffill()
    # returns_data_bom = returns_data.resample('MS').first()

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
    prev_date=0
    for date in returns_data.index:
        # print(date, date.month)
        if date.month == 1 and date.year <= 2021 and date.year != prev_date:  # and date.day == 1:  # Check if the date is January 1st
            # print(date, date.month,"\n --------")
            new_date = pd.Timestamp(date).replace(day=1)
            current_weights = weights_monthly.loc[new_date]


        # Adjust weights dynamically based on the previous weights and returns
        aligned_weights = current_weights.reindex(returns_data.columns).fillna(0)
        aligned_returns = returns_data.loc[date].reindex(aligned_weights.index).fillna(0)

        # Calculate portfolio value
        portfolio_value = (current_weights * (1 + returns_data.loc[date].fillna(0))).sum()

        # Update weights dynamically
        current_weights = (current_weights * (1 + returns_data.loc[date].fillna(0))) / portfolio_value

        dynamic_weights.loc[date] = current_weights
        prev_date = date.year

    dynamic_weights.ffill(inplace=True)

    # Use select_slider for available dates
    available_dates = dynamic_weights.index[dynamic_weights.index >= pd.Timestamp('2007-01-01')].to_pydatetime()
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


# ***********************************************************************************************************
# Performance
# ***********************************************************************************************************
if choice == "Performance":
    st.title("Performance")

    list_portfolios = [
        'equity_amer', 'equity_em', 'equity_eur', 'equity_pac',
        'metals', 'commodities', 'volatilities', 'crypto', 'erc'
    ]

    # Initialize session state for selected portfolios
    if "selected_portfolios" not in st.session_state:
        st.session_state["selected_portfolios"] = []

    # Button to display all portfolios
    if st.button("Display All"):
        st.session_state["selected_portfolios"] = list_portfolios

    # Portfolio Selection
    st.subheader("Select portfolios to display", divider="gray")
    rows = [list_portfolios[i:i + 4] for i in range(0, len(list_portfolios), 4)]
    for row in rows:
        cols = st.columns(len(row))
        for col, portfolio in zip(cols, row):
            with col:
                toggle = st.checkbox(
                    portfolio,
                    value=portfolio in st.session_state["selected_portfolios"]
                )
                if toggle and portfolio not in st.session_state["selected_portfolios"]:
                    st.session_state["selected_portfolios"].append(portfolio)
                elif not toggle and portfolio in st.session_state["selected_portfolios"]:
                    st.session_state["selected_portfolios"].remove(portfolio)

    selected_portfolios = st.session_state["selected_portfolios"]

    if selected_portfolios:
        # Gamma slider
        session_gamma = st.session_state.get('gamma_value')  # Default gamma if not set
        gamma_value = st.slider(
            "Gamma Value",
            min_value=0.0,
            max_value=2.0,
            step=0.01,
            value=session_gamma,
            help="Adjust the risk preference using the gamma slider."
        )

        # Save the gamma value in session state
        gamma_value = get_nearest_gamma(gamma_value)
        st.session_state["gamma_value"] = gamma_value

        # Load portfolio returns
        # portfolio_returns = load_portfolio_returns()

        #     master_df = load_master_df()
        #     corr_matrix, _, _, _ = load_correlation_and_returns(master_df)
        #     # del master_df, _
        #     frontier_data = load_efficient_frontier_data()
        #     available_dates = extract_available_dates(frontier_data)
        #     gamma_array = frontier_data.index.get_level_values('gamma').unique().values
        #     # del frontier_data
        portfolio_returns = optimize_floats(load_portfolio_returns(), robust=True)

        # Get returns for the selected gamma
        try:
            returns_gamma = portfolio_returns.xs(gamma_value, level='gamma')
            del portfolio_returns
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
                "Max Drawdown": True,
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
