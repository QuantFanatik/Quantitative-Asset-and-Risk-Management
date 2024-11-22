import streamlit as st
import numpy as np
from scipy.spatial import distance
from scipy.interpolate import griddata
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os
from scipy.interpolate import splprep, splev

# Data Source 1
# ---------------------------------------------------------------------------------------
root = os.path.dirname(__file__)
returns_path = os.path.join(root, 'data', 'portfolio_returns.csv')
weights_path = os.path.join(root, 'data', 'portfolio_weights.csv')

portfolio_returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
portfolio_weights = pd.read_csv(weights_path, index_col=0, parse_dates=True, header=[0, 1])
portfolio_weights = portfolio_weights[portfolio_weights.index.year >= 2006]

# Data Source 2
# ---------------------------------------------------------------------------------------
dir_path = os.path.dirname(os.path.realpath(__file__))
master_path = os.path.join(dir_path, "data/data_YF/master_df.csv")
master_df = pd.read_csv(master_path, index_col=0, parse_dates=True)

# Help for making the web clean
# --------------------------------------------------------------------------------------
list_type_portfolio = ['equity_amer', 'equity_em', 'equity_eur', 'equity_pac',
                       'metals', 'commodities', 'crypto', 'volatilities']

list_clean_name = ['Metals', 'Commodities', 'Crypto', 'Volatilities',
                   'North American Equities', 'Emerging Markets Equities', 'European Equities', 'Asia-Pacific Equities']

list_commodities = ["Lean_Hogs", "Crude_Oil", "Live_Cattle", "Soybeans", "Wheat", "Corn", "Natural_Gas"]
list_crypto = ["Bitcoin", "Ethereum"]
list_metals = ["Gold", "Platinum", "Palladium", "Silver", "Copper"]
list_volatilities = ["Russell_2000_RVX", "VVIX_VIX_of_VIX", "MOVE_bond_market_volatility",
                     "VXO-S&P_100_volatility", "Nasdaq_VXN", "VIX"]

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

# Efficient Frontier Functions
# ---------------------------------------------------------------------------------------
@st.cache_data
def load_efficient_frontier_data():
    directory = os.path.join(root, 'data')
    base_filename = 'efficient_frontiers'
    chunk_files = sorted(
        [os.path.join(directory, f) for f in os.listdir(directory)
         if f.startswith(base_filename) and f.endswith('.csv')]
    )
    if not chunk_files:
        st.error(f"No chunk files found for base filename '{base_filename}' in '{directory}'")
        return pd.DataFrame()
    all_chunks = [pd.read_csv(chunk, header=[0, 1], index_col=[0, 1, 2]) for chunk in chunk_files]
    data = pd.concat(all_chunks, axis=0)

    # Adjust columns
    new_columns = [(top, "" if "Unnamed" in bottom else bottom) for top, bottom in data.columns]
    data.columns = pd.MultiIndex.from_tuples(new_columns, names=["category", "attribute"])

    # Ensure the index levels are named correctly
    data.index.set_names(["year", "gamma", "portfolio"], inplace=True)
    return data

# Plot Efficient Frontier Functions
# ---------------------------------------------------------------------------------------

def plot_efficient_frontier(data, selected_portfolio, selected_year):
    # Filter data for the selected portfolio and year
    try:
        data_to_plot = data.loc[(selected_year, slice(None), selected_portfolio), :]
    except KeyError:
        return None  # Return None if data is not available for the selection

    # Extract variance, expected return, and gamma
    x = data_to_plot[('metrics', 'expected_variance')].values
    y = data_to_plot[('metrics', 'expected_return')].values
    gamma_values = data_to_plot.index.get_level_values('gamma').values

    # Combine x, y, and gamma into points
    points = np.column_stack((x, y, gamma_values))

    # Sort points based on proximity (distance-based sorting)
    sorted_points = [points[0]]  # Start with the first point
    remaining_points = points[1:]

    while len(remaining_points) > 0:
        last_point = sorted_points[-1][:2]  # Only x and y for distance calculation
        distances = distance.cdist([last_point], remaining_points[:, :2], metric="euclidean").flatten()
        nearest_idx = np.argmin(distances)
        sorted_points.append(remaining_points[nearest_idx])
        remaining_points = np.delete(remaining_points, nearest_idx, axis=0)

    sorted_points = np.array(sorted_points)
    x_sorted, y_sorted, gamma_sorted = sorted_points[:, 0], sorted_points[:, 1], sorted_points[:, 2]

    # Create the efficient frontier plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_sorted,
        y=y_sorted,
        mode='lines+markers',
        name=f"{selected_portfolio.capitalize()} ({selected_year})",
        hovertemplate="Variance: %{x}<br>Expected Return: %{y}<br>Gamma: %{customdata}",
        customdata=gamma_sorted
    ))

    # Plot the red dot at the user's gamma
    gamma_value = st.session_state.get('gamma_value', None)
    if gamma_value is not None:
        # Find the closest gamma in the data
        closest_gamma_idx = (np.abs(gamma_sorted - gamma_value)).argmin()
        x_gamma = x_sorted[closest_gamma_idx]
        y_gamma = y_sorted[closest_gamma_idx]
        fig.add_trace(go.Scatter(
            x=[x_gamma],
            y=[y_gamma],
            mode='markers',
            marker=dict(color='red', size=10),
            name=f"Your Portfolio (Gamma={gamma_value:.2f})",
            hovertemplate="Variance: %{x}<br>Expected Return: %{y}<br>Gamma: %{customdata}",
            customdata=[gamma_sorted[closest_gamma_idx]]
        ))
    else:
        st.warning("Please set your Gamma in the 'Risk Profiling' section.")

    fig.update_layout(
        title=f"Efficient Frontier for {selected_portfolio.capitalize()} in Year {selected_year}",
        xaxis_title="Variance (Risk)",
        yaxis_title="Expected Return",
        width=800,
        height=600
    )
    return fig

# Updated Plot Efficient Frontier 3D Function
# ---------------------------------------------------------------------------------------

def plot_efficient_frontier_3d(data, selected_portfolio):
    # Filter data for the selected portfolio
    try:
        data_to_plot = data.xs(selected_portfolio, level="portfolio")
    except KeyError:
        return None  # Return None if data is not available for the selection

    # Extract variance, expected return, gamma, and year
    x = data_to_plot[('metrics', 'expected_variance')].values
    y = data_to_plot[('metrics', 'expected_return')].values
    gamma_values = data_to_plot.index.get_level_values('gamma').values
    years = data_to_plot.index.get_level_values('year').values

    # Combine x, y, gamma, and year into points
    points = np.column_stack((x, y, gamma_values, years))

    # Prepare data for surface plot
    df_surface = pd.DataFrame(points, columns=['Variance', 'Return', 'Gamma', 'Year'])

    # Create grid for surface
    grid_x, grid_y = np.meshgrid(
        np.linspace(df_surface['Variance'].min(), df_surface['Variance'].max(), 50),
        np.linspace(df_surface['Return'].min(), df_surface['Return'].max(), 50)
    )

    # Interpolate Z values (Year) over the grid
    grid_z = griddata(
        points=(df_surface['Variance'], df_surface['Return']),
        values=df_surface['Year'],
        xi=(grid_x, grid_y),
        method='linear'
    )

    # Create the figure
    fig = go.Figure()

    # Add the surface
    fig.add_trace(go.Surface(
        x=grid_x,
        y=grid_y,
        z=grid_z,
        colorscale='Viridis',
        opacity=0.7,
        name='Efficient Frontier Surface'
    ))

    # Plot the investor's gamma over time
    gamma_value = st.session_state.get('gamma_value', None)
    if gamma_value is not None:
        # Find points corresponding to the investor's gamma
        gamma_mask = np.isclose(df_surface['Gamma'], gamma_value, atol=1e-2)
        gamma_data = df_surface[gamma_mask]
        if not gamma_data.empty:
            x_gamma = gamma_data['Variance'].values
            y_gamma = gamma_data['Return'].values
            z_gamma = gamma_data['Year'].values

            # Sort the data by Year for plotting
            sorted_indices = np.argsort(z_gamma)
            x_gamma_sorted = x_gamma[sorted_indices]
            y_gamma_sorted = y_gamma[sorted_indices]
            z_gamma_sorted = z_gamma[sorted_indices]

            # Add the red line or dots
            fig.add_trace(go.Scatter3d(
                x=x_gamma_sorted,
                y=y_gamma_sorted,
                z=z_gamma_sorted,
                mode='lines+markers',
                marker=dict(color='red', size=5),
                line=dict(color='red', width=2),
                name=f"Your Portfolio (Gamma={gamma_value:.2f})",
                hovertemplate='Variance: %{x}<br>Expected Return: %{y}<br>Year: %{z}'
            ))
        else:
            st.warning("No data available for your Gamma value on the 3D plot.")
    else:
        st.warning("Please set your Gamma in the 'Risk Profiling' section.")

    # Update layout
    fig.update_layout(
        title=f"3D Efficient Frontier for {selected_portfolio.capitalize()} Over Time",
        scene=dict(
            xaxis_title='Variance (Risk)',
            yaxis_title='Expected Return',
            zaxis_title='Year',
        ),
        width=800,
        height=600
    )

    return fig

# ***********************************************************************************************************
# Principal bar
# ***********************************************************************************************************
with st.sidebar:
    st.title("Portfolio Optimization")
    # Added "Risk Profiling" to the steps
    choice = st.radio("Steps", ["Risk Profiling", "Data Exploration", "Sub-Portfolio", "Optimal portfolio", "Performance"])
    st.info("This tool uses equal risk contributions method to select optimal weights for each "
            "type of asset classes. Then, we use Markowitz optimization to choose an optimal portfolio.")

# ***********************************************************************************************************
# Risk Profiling
# ***********************************************************************************************************
if choice == "Risk Profiling":

    st.title("Risk Profiling")
    gamma_known = st.radio("Do you know your Gamma?", ("Yes", "No"))

    if gamma_known == "Yes":
        gamma_value = st.number_input("Please enter your Gamma value", min_value=0.0, max_value=None, value=1.0)
        st.write(f"You have entered Gamma = {gamma_value}")
        st.session_state['gamma_value'] = gamma_value
    else:
        st.write("We will ask you some questions to help determine your Gamma.")
        # Questions to define the Gamma
        q1 = st.slider("On a scale from 1 (Very low risk tolerance) to 5 (Very high risk tolerance), how would you rate your risk tolerance?", 1, 5, 3)
        q2 = st.selectbox("What is your investment horizon?", ["Short-term (less than 3 years)", "Medium-term (3-7 years)", "Long-term (more than 7 years)"])
        q3 = st.selectbox("What is your primary investment goal?", ["Capital preservation", "Income generation", "Growth"])

        # Assign numerical values to the answers
        risk_tolerance_score = q1

        if q2 == "Short-term (less than 3 years)":
            horizon_score = 1
        elif q2 == "Medium-term (3-7 years)":
            horizon_score = 2
        else:
            horizon_score = 3

        if q3 == "Capital preservation":
            goal_score = 1
        elif q3 == "Income generation":
            goal_score = 2
        else:
            goal_score = 3

        # Calculate Gamma (this is a simplified example; adjust as needed)
        gamma_score = risk_tolerance_score + horizon_score + goal_score
        gamma_value = 10 / gamma_score  # Higher score implies lower gamma

        st.write(f"Based on your answers, your estimated Gamma is **{gamma_value:.2f}**")
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
        st.subheader("Expected Returns and Volatilities", divider="gray")
        st.write("Expected Returns")
        st.bar_chart(data_use.mean())
        st.write("Expected Volatilities")
        st.bar_chart(data_use.std())

        # Heatmap visualization
        st.subheader("Correlation Heatmap", divider="gray")
        st.write("")
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
        if frontier_data.empty:
            st.error("Efficient frontier data is unavailable.")
        else:
            # Add option to select plot type
            plot_type = st.radio("Select Plot Type", ["Year by Year", "3D over Time"])

            if plot_type == "Year by Year":
                # Get available years for the selected portfolio
                try:
                    gamma_value = st.session_state.get('gamma_value', None)
                    if gamma_value is None:
                        st.warning("Please set your gamma in the 'Risk Profiling' section.")
                    else:
                        gamma_values_in_data = frontier_data.index.get_level_values('gamma').unique()
                        closest_gamma = gamma_values_in_data[np.abs(gamma_values_in_data - gamma_value).argmin()]
                        portfolio_years = frontier_data.xs((closest_gamma, selected_portfolio), level=("gamma", "portfolio")).index.get_level_values("year").unique()
                        selected_year = st.slider(
                            "Select Year",
                            int(portfolio_years.min()),
                            int(portfolio_years.max()),
                            int(portfolio_years.min()),
                            step=1
                        )
                        fig_ef = plot_efficient_frontier(frontier_data, selected_portfolio, selected_year)
                        if fig_ef is not None:
                            st.plotly_chart(fig_ef)
                        else:
                            st.error(f"No data available for {selection} in year {selected_year}")
                except KeyError:
                    st.error(f"No data available for {selection}")
                    st.stop()
            else:
                # Plot 3D efficient frontier over time
                fig_ef_3d = plot_efficient_frontier_3d(frontier_data, selected_portfolio)
                if fig_ef_3d is not None:
                    st.plotly_chart(fig_ef_3d)
                else:
                    st.error(f"No data available for {selection}")

    elif selection in ['North American Equities', 'Emerging Markets Equities', 'European Equities',
                       'Asia-Pacific Equities']:

        st.info("Data exploration is different for equities due to the large number of securities.")

        if selection == "North American Equities":
            mean = str(round(float(portfolio_returns["equity_amer"].mean() * 100), 2))
            vol = str(round(float(portfolio_returns["equity_amer"].std() * 100), 2))
            nb_eq = int(list_data_equity_amer.shape[0])
            list_data_equity_amer.rename(columns={"equity_amer": "ISIN"}, inplace=True)
            list_isin = list_data_equity_amer

        elif selection == "Emerging Markets Equities":
            mean = str(round(float(portfolio_returns["equity_em"].mean() * 100), 2))
            vol = str(round(float(portfolio_returns["equity_em"].std() * 100), 2))
            nb_eq = int(list_data_equity_em.shape[0])
            list_isin = list_data_equity_em

        elif selection == "European Equities":
            mean = str(round(float(portfolio_returns["equity_eur"].mean() * 100), 2))
            vol = str(round(float(portfolio_returns["equity_eur"].std() * 100), 2))
            nb_eq = int(list_data_equity_eur.shape[0])
            list_isin = list_data_equity_eur

        elif selection == "Asia-Pacific Equities":
            mean = str(round(float(portfolio_returns["equity_pac"].mean() * 100), 2))
            vol = str(round(float(portfolio_returns["equity_pac"].std() * 100), 2))
            nb_eq = int(list_data_equity_pac.shape[0])
            list_isin = list_data_equity_pac

        # Display expected return, volatility, and equity details
        st.subheader("Expected Return and Volatility", divider="gray")
        st.markdown(f"**Expected Return**: {mean}%")
        st.markdown(f"**Expected Volatility**: {vol}%")
        st.write("")
        st.subheader("Additional Information", divider="gray")
        st.markdown(f"**Number of Equities**: {nb_eq}")
        st.write("")
        st.markdown("**Equities Composition**:")
        st.write(list_isin)

        # Efficient Frontier for equity portfolios
        st.subheader("Efficient Frontier")
        frontier_data = load_efficient_frontier_data()
        if frontier_data.empty:
            st.error("Efficient frontier data is unavailable.")
        else:
            # Add option to select plot type
            plot_type = st.radio("Select Plot Type", ["Year by Year", "3D over Time"])

            if plot_type == "Year by Year":
                try:
                    gamma_value = st.session_state.get('gamma_value', None)
                    if gamma_value is None:
                        st.warning("Please set your gamma in the 'Risk Profiling' section.")
                    else:
                        gamma_values_in_data = frontier_data.index.get_level_values('gamma').unique()
                        closest_gamma = gamma_values_in_data[np.abs(gamma_values_in_data - gamma_value).argmin()]
                        portfolio_years = frontier_data.xs((closest_gamma, selected_portfolio), level=("gamma", "portfolio")).index.get_level_values("year").unique()
                        selected_year = st.slider(
                            "Select Year",
                            int(portfolio_years.min()),
                            int(portfolio_years.max()),
                            int(portfolio_years.min()),
                            step=1
                        )
                        fig_ef = plot_efficient_frontier(frontier_data, selected_portfolio, selected_year)
                        if fig_ef is not None:
                            st.plotly_chart(fig_ef)
                        else:
                            st.error(f"No data available for {selection} in year {selected_year}")
                except KeyError:
                    st.error(f"No data available for {selection}")
                    st.stop()
            else:
                fig_ef_3d = plot_efficient_frontier_3d(frontier_data, selected_portfolio)
                if fig_ef_3d is not None:
                    st.plotly_chart(fig_ef_3d)
                else:
                    st.error(f"No data available for {selection}")

# Rest of your code remains unchanged...
# ***********************************************************************************************************
# Sub-portfolio Analysis
# ***********************************************************************************************************
if choice == "Sub-Portfolio":

    st.title("Sub-Portfolio")
    selection = st.selectbox("Choose portfolio class", list_clean_name, index=0)
    st.info("We have to optimize each class of portfolio before using Markowitz in our global portfolio")

    if selection == "Commodities":
        data_use = portfolio_weights["commodities"]

    if selection == "Metals":
        data_use = portfolio_weights["metals"]

    if selection == "Crypto":
        data_use = portfolio_weights[portfolio_weights.index >= pd.to_datetime("2014-12-31")]["crypto"]

    if selection == "Volatilities":
        data_use = portfolio_weights["volatilities"]

    if selection == "North American Equities":
        data_use = portfolio_weights["equity_amer"]

    if selection == "Emerging Markets Equities":
        data_use = portfolio_weights["equity_em"]

    if selection == "European Equities":
        data_use = portfolio_weights["equity_eur"]

    if selection == "Asia-Pacific Equities":
        data_use = portfolio_weights["equity_pac"]

    # Date slider
    # -----------------------------------------------------------------------------------------
    if selection in ["Commodities", "Metals", "Crypto", "Volatilities"]:
        data_use.index = pd.to_datetime(data_use.index)

        date_slider = st.slider(
            'Choose the date',
            min_value=data_use.index.min().to_pydatetime(),
            max_value=data_use.index.max().to_pydatetime(),
            format="YYYY-MM-DD",
            value=data_use.index.min().to_pydatetime()
        )

        # Convert the selected date slider value to pandas Timestamp
        date_slider = pd.Timestamp(date_slider)

        # Filter data up to selected date
        filtered_data = data_use[data_use.index <= date_slider]

        # Extract weights for the selected date
        latest_data = filtered_data.iloc[-1]

        # Do the pie chart
        fig = px.pie(latest_data, values=latest_data.values, names=latest_data.index)

        st.subheader(f"Weight allocation on {date_slider.strftime('%Y-%m-%d')}")
        st.plotly_chart(fig)
    # -----------------------------------------------------------------------------------------
    st.subheader("Weight allocation over time")
    st.bar_chart(data_use)

# ***********************************************************************************************************
# Optimal portfolio
# ***********************************************************************************************************
if choice == "Optimal portfolio":

    st.title("Optimal portfolio")
    selection = st.radio("Choose visualization", ["Efficient frontier", "Portfolio composition"])

    if selection == "Efficient frontier":
        st.info("Efficient frontier visualization is under development.")

    if selection == "Portfolio composition":

        data_use = portfolio_weights["erc"]
        data_use.index = pd.to_datetime(data_use.index)

        date_slider = st.slider(
            'Choose the date',
            min_value=data_use.index.min().to_pydatetime(),
            max_value=data_use.index.max().to_pydatetime(),
            format="YYYY-MM-DD",
            value=data_use.index.min().to_pydatetime()
        )

        # Convert the selected date slider value to pandas Timestamp
        date_slider = pd.Timestamp(date_slider)

        # Filter data up to selected date
        filtered_data = data_use[data_use.index <= date_slider]

        # Extract weights for the selected date
        latest_data = filtered_data.iloc[-1]

        # Do the pie chart
        fig = px.pie(latest_data, values=latest_data.values, names=latest_data.index)

        st.subheader(f"Weight allocation on {date_slider.strftime('%Y-%m-%d')}")
        st.plotly_chart(fig)

        st.subheader("Weight allocation over time")
        st.bar_chart(data_use)

# ***********************************************************************************************************
# Performance
# ***********************************************************************************************************
if choice == "Performance":
    st.title("Performance")
    st.subheader("Cumulative Log Return and Drawdown", divider="gray")

    # Portfolio selection with toggle switches arranged horizontally
    list_portfolios = [
        'equity_amer', 'equity_em', 'equity_eur', 'equity_pac',
        'metals', 'commodities', 'volatilities', 'crypto', 'erc'
    ]
    selected_portfolios = []

    st.write("Select portfolios to display:")

    # Split into rows of 4 columns each
    rows = [list_portfolios[i:i + 4] for i in range(0, len(list_portfolios), 4)]
    for row in rows:
        cols = st.columns(len(row))
        for col, portfolio in zip(cols, row):
            with col:
                toggle = st.checkbox(portfolio, value=False)
                if toggle:
                    selected_portfolios.append(portfolio)

    # "Display All" button
    if st.button("Display All"):
        selected_portfolios = list_portfolios

    if selected_portfolios:
        # Calculate cumulative log returns for selected portfolios
        log_returns_selected = np.log(1 + portfolio_returns[selected_portfolios])
        cumulative_log_returns_selected = log_returns_selected.cumsum() + 1
        cumulative_log_returns_selected = cumulative_log_returns_selected[
            cumulative_log_returns_selected.index >= '2006-01-01'
        ]

        # Plot cumulative log returns
        st.write("### Cumulative Log Return")
        st.line_chart(
            cumulative_log_returns_selected,
            x_label="Year",
            y_label="Cumulative Log Return",
            height=400,
            width=700,
            use_container_width=True
        )

        # Calculate and plot drawdown
        st.write("### Drawdown")
        drawdowns = (cumulative_log_returns_selected - cumulative_log_returns_selected.cummax()) / cumulative_log_returns_selected.cummax()
        st.line_chart(
            drawdowns,
            x_label="Year",
            y_label="Drawdown",
            height=400,
            width=700,
            use_container_width=True
        )

        # Calculate performance metrics for each portfolio
        metrics_data = {}
        for portfolio in selected_portfolios:
            returns = portfolio_returns[portfolio]
            cumulative_returns = np.log(1 + returns).cumsum()

            mean_return = returns.mean()
            volatility = returns.std()
            sharpe_ratio = mean_return / volatility if volatility != 0 else 0
            max_drawdown = (cumulative_returns - cumulative_returns.cummax()).min()
            drawdown_duration = (cumulative_returns - cumulative_returns.cummax()).idxmin()

            metrics_data[portfolio] = {
                "Mean Return": mean_return,
                "Volatility": volatility,
                "Sharpe Ratio": sharpe_ratio,
                "Max Drawdown": max_drawdown,
                "Max Drawdown Duration": drawdown_duration.strftime('%Y-%m-%d'),
            }

        # Create DataFrame for metrics
        metrics_df = pd.DataFrame(metrics_data).T
        metrics_df.index.name = "Portfolio"

        # Identify numeric columns for styling
        numeric_columns = ["Mean Return", "Volatility", "Sharpe Ratio", "Max Drawdown"]

        # Create mapping for whether higher is better
        better_high = {
            "Mean Return": True,
            "Volatility": False,
            "Sharpe Ratio": True,
            "Max Drawdown": False,
        }

        # Define the highlight function per column
        def highlight_metrics_column(s):
            """
            Highlight function to color values between min and max in each column.
            Green for better, red for worse.
            """
            is_better_high = better_high.get(s.name, True)
            min_val = s.min()
            max_val = s.max()
            if min_val == max_val:
                normalized = s * 0.0  # All zeros if min and max are equal
            else:
                normalized = (s - min_val) / (max_val - min_val)  # Normalize between 0 and 1
            if not is_better_high:
                # Reverse the normalized values for metrics where lower is better
                normalized = 1 - normalized
            # Generate color gradients
            colors = [
                f"background-color: rgba({255 - int(255 * x)}, {int(255 * x)}, 0, 0.8)"
                for x in normalized
            ]
            return colors

        # Create a Styler object and apply formatting and styling
        styled_metrics = metrics_df.style.apply(
            highlight_metrics_column, subset=numeric_columns, axis=0
        ).format({
            "Mean Return": "{:.2%}",
            "Volatility": "{:.2%}",
            "Sharpe Ratio": "{:.2f}",
            "Max Drawdown": "{:.2%}",
        })

        # Display the styled DataFrame
        st.write("### Portfolio Metrics")
        st.write(styled_metrics)

        st.write(
            "Cumulative log returns, drawdowns, and performance metrics for the selected portfolios are displayed above."
        )
    else:
        st.warning("Please select at least one portfolio to view its performance or click 'Display All'.")

    st.write("Do you want to invest in our strategy?")
    sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
    selected = st.feedback("thumbs")
    if selected is not None:
        st.markdown(f"You selected: {sentiment_mapping[selected]}")

# streamlit run your_script.py
