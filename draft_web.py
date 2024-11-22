import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os


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
    files = [f for f in os.listdir('data') if f.startswith("efficient_frontiers_") and f.endswith(".hdf")]
    data_chunks = []

    if not files:
        return pd.DataFrame()  # Return empty DataFrame if no files found

    for file in files:
        file_path = os.path.join('data', file)
        try:
            # Read the HDF5 file
            chunk = pd.read_hdf(file_path)

            # Add `year` if not already present
            if 'year' not in chunk.index.names:
                year = int(file.split('_')[-1].split('.')[0])
                chunk['year'] = year
                chunk = chunk.set_index('year', append=True)

            data_chunks.append(chunk)
        except Exception as e:
            st.warning(f"Could not load file {file}: {e}")

    # Concatenate all chunks, handle empty chunks
    if data_chunks:
        return pd.concat(data_chunks).sort_index()
    else:
        return pd.DataFrame()


@st.cache_data
def plot_efficient_frontiers_with_slider(data):
    if data.empty:
        # Return an empty Plotly figure with a placeholder message
        fig = go.Figure()
        fig.update_layout(
            title="No data available for plotting efficient frontiers",
            xaxis_title="Variance (Risk)",
            yaxis_title="Expected Return",
            width=800,
            height=600
        )
        return fig

    years = data.index.get_level_values("year").unique()
    portfolios = data.index.get_level_values("portfolio").unique()

    fig = go.Figure()

    for year in years:
        yearly_data = data.xs(year, level="year")
        for portfolio in portfolios:
            if portfolio not in yearly_data.index.get_level_values("portfolio").unique():
                continue
            portfolio_data = yearly_data.xs(portfolio, level="portfolio")
            x = portfolio_data[("metrics", "expected_variance")]
            y = portfolio_data[("metrics", "expected_return")]
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines+markers',
                name=f"{portfolio} ({year})",
                visible=(year == years[0]),
                legendgroup=str(year),
                hovertemplate=f"<b>Portfolio:</b> {portfolio}<br><b>Year:</b> {year}<br>Expected Return: %{{y}}<br>Variance: %{{x}}"
            ))

    steps = []
    for i, year in enumerate(years):
        step = dict(
            method="restyle",
            args=["visible", [False] * len(fig.data)],
            label=str(year),
        )
        for j in range(i * len(portfolios), (i + 1) * len(portfolios)):
            step["args"][1][j] = True
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Year: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        title="Efficient Frontiers for All Portfolios Over Time",
        xaxis_title="Variance (Risk)",
        yaxis_title="Expected Return",
        sliders=sliders,
        xaxis_range=[0, 0.05],
        yaxis_range=[-0.1, 0.2],
        width=800,
        height=600
    )
    return fig


#***********************************************************************************************************
# Principal bar
#***********************************************************************************************************
with st.sidebar:
    st.title("Portfolio Optimization")
    choice = st.radio("Steps", ["Data Exploration", "Equal Risk Contributions", "Optimal portfolio", "Performance"])
    st.info("This tool uses equal risk contributions method to select optimal weights for each "
            "type of asset classes. Then, we use Markowitz optimization to choose an optimal portfolio.")

#***********************************************************************************************************
# Data Exploration
#***********************************************************************************************************
if choice == "Data Exploration":

    st.title("Data Exploration")
    selection = st.selectbox("Choose portfolio class", list_clean_name, index=0)

    if selection in ["Commodities", "Metals", "Crypto", "Volatilities"]:

        if selection == "Commodities":
            data_use = df_commodities
            correl_matrix = corr_matrix.loc[list_commodities, list_commodities]

        if selection == "Metals":
            data_use = df_metals
            correl_matrix = corr_matrix.loc[list_metals, list_metals]

        if selection == "Crypto":
            data_use = df_crypto
            correl_matrix = corr_matrix.loc[list_crypto, list_crypto]

        if selection == "Volatilities":
            data_use = df_volatilities
            correl_matrix = corr_matrix.loc[list_volatilities, list_volatilities]

        # Display expected returns and volatilities
        st.subheader("Expected returns and volatilities", divider="gray")
        st.write("Expected returns")
        st.bar_chart(data_use.mean())
        st.write("Expected volatilities")
        st.bar_chart(data_use.std())

        # Heatmap visualization
        st.subheader("Heatmap", divider="gray")
        st.write("")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correl_matrix, annot=True, cmap="coolwarm", ax=ax)
        fig.patch.set_alpha(0)  # Transparent background
        ax.patch.set_alpha(0)  # Transparent axes background
        st.pyplot(fig)

        # Efficient Frontier for the selected portfolio
        st.subheader("Efficient Frontier")
        frontier_data = load_efficient_frontier_data()
        if frontier_data.empty:
            st.error("Efficient frontier data is unavailable.")
        else:
            fig_ef = plot_efficient_frontiers_with_slider(frontier_data)
            st.plotly_chart(fig_ef)

    elif selection in ['North American Equities', 'Emerging Markets Equities', 'European Equities',
                       'Asia-Pacific Equities']:

        st.info("The data exploration is different for equities because of the large number of securities.")

        if selection == "North American Equities":
            mean = str(round(float(portfolio_returns["equity_amer"].mean() * 100), 2))
            vol = str(round(float(portfolio_returns["equity_amer"].std() * 100), 2))
            nb_eq = int(list_data_equity_amer.shape[0])
            list_data_equity_amer.rename(columns={"equity_amer": "ISIN"}, inplace=True)
            list_isin = list_data_equity_amer

        if selection == "Emerging Markets Equities":
            mean = str(round(float(portfolio_returns["equity_em"].mean() * 100), 2))
            vol = str(round(float(portfolio_returns["equity_em"].std() * 100), 2))
            nb_eq = int(list_data_equity_em.shape[0])
            list_isin = list_data_equity_em

        if selection == "European Equities":
            mean = str(round(float(portfolio_returns["equity_eur"].mean() * 100), 2))
            vol = str(round(float(portfolio_returns["equity_eur"].std() * 100), 2))
            nb_eq = int(list_data_equity_eur.shape[0])
            list_isin = list_data_equity_eur

        if selection == "Asia-Pacific Equities":
            mean = str(round(float(portfolio_returns["equity_pac"].mean() * 100), 2))
            vol = str(round(float(portfolio_returns["equity_pac"].std() * 100), 2))
            nb_eq = int(list_data_equity_pac.shape[0])
            list_isin = list_data_equity_pac

        # Display expected return, volatility, and equity details
        st.subheader("Expected return and volatility", divider="gray")
        st.markdown(f"**Expected return**: {mean}%")
        st.markdown(f"**Expected volatility**: {vol}%")
        st.write("")
        st.subheader("Additional Information", divider="gray")
        st.markdown(f"**Number of equities**: {nb_eq}")
        st.write("")
        st.markdown("**Equities composition**:")
        st.write(list_isin)

        # Efficient Frontier for equity portfolios
        st.subheader("Efficient Frontier for Equity Portfolios")
        frontier_data = load_efficient_frontier_data()
        if frontier_data.empty:
            st.error("Efficient frontier data is unavailable.")
        else:
            fig_ef = plot_efficient_frontiers_with_slider(frontier_data)
            st.plotly_chart(fig_ef)



#***********************************************************************************************************
# Equal Risk Contributions
#***********************************************************************************************************
if choice == "Equal Risk Contributions":

    st.title("Equal Risk Contributions")
    selection = st.selectbox("Choose portfolio class", list_clean_name, index=0)
    st.info("We have to optimize each class of portfolio before use Matkowitz in our global porfolio")

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
            min_value=data_use.index.min().to_pydatetime(),  # Convert to Python datetime
            max_value=data_use.index.max().to_pydatetime(),  # Convert to Python datetime
            format="YYYY-MM-DD",
            value=data_use.index.min().to_pydatetime()  # Use the first date as default
        )

        # Convert the selected date slider value to pandas Timestamp
        date_slider = pd.Timestamp(date_slider)  # Ensure it's a pandas Timestamp for consistency

        # Filter data up to selected date
        filtered_data = data_use[data_use.index <= date_slider]

        # Extract weights for the selected date
        latest_data = filtered_data.iloc[-1]  # Prend la dernière ligne filtrée

        # Do the pie chart
        fig = px.pie(latest_data, values=latest_data.values, names=latest_data.index)

        st.subheader(f"Weight allocation in {date_slider.strftime('%Y-%m-%d')}")
        st.plotly_chart(fig)
    # -----------------------------------------------------------------------------------------
    st.subheader("Weight allocation over time")
    st.bar_chart(data_use)

#***********************************************************************************************************
# Optimal portfolio
#***********************************************************************************************************
if choice == "Optimal portfolio":

    st.title("Optimal portfolio")
    selection = st.radio("Choose visualization",["Efficient frontier","Portfolio composition"])

    if selection == "Efficient frontier":
        pass

    if selection == "Portfolio composition":

        data_use = portfolio_weights["erc"]
        data_use.index = pd.to_datetime(data_use.index)

        date_slider = st.slider(
            'Choose the date',
            min_value=data_use.index.min().to_pydatetime(),  # Convert to Python datetime
            max_value=data_use.index.max().to_pydatetime(),  # Convert to Python datetime
            format="YYYY-MM-DD",
            value=data_use.index.min().to_pydatetime()  # Use the first date as default
        )

        # Convert the selected date slider value to pandas Timestamp
        date_slider = pd.Timestamp(date_slider)  # Ensure it's a pandas Timestamp for consistency

        # Filter data up to selected date
        filtered_data = data_use[data_use.index <= date_slider]

        # Extract weights for the selected date
        latest_data = filtered_data.iloc[-1]  # Prend la dernière ligne filtrée

        # Do the pie chart
        fig = px.pie(latest_data, values=latest_data.values, names=latest_data.index)

        st.subheader(f"Weight allocation in {date_slider.strftime('%Y-%m-%d')}")
        st.plotly_chart(fig)

        st.subheader("Weight allocation over time")
        st.bar_chart(data_use)

# ***********************************************************************************************************
# Performance
# ***********************************************************************************************************
if choice == "Performance":

    st.title("Performance")
    st.subheader("Cumulative Log Return", divider="gray")

    choice = st.toggle("Our strategy", value=True)

    if choice == True:
        # Calculate cumulative log returns
        log_returns = np.log(1 + portfolio_returns["erc"])
        cumulative_log_returns = log_returns.cumsum()
        cumulative_log_returns = cumulative_log_returns[cumulative_log_returns.index >= '2006-01-01']

        # Plot cumulative log returns
        st.line_chart(cumulative_log_returns, x_label="Year", y_label="Cumulative Log Return", height=400, width=700,
                      use_container_width=False)

        st.write("Do you want to invest in our strategy?")
        sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
        selected = st.feedback("thumbs")
        if selected is not None:
            st.markdown(f"You selected: {sentiment_mapping[selected]}")

    if choice == False:
        list_ = ['equity_amer', 'equity_em', 'equity_eur', 'equity_pac', 'metals', 'commodities', 'volatilities', 'crypto', 'erc']

        # Calculate cumulative log returns for multiple asset classes
        log_returns_rest = np.log(1 + portfolio_returns[list_])
        cumulative_log_returns_rest = log_returns_rest.cumsum()
        cumulative_log_returns_rest = cumulative_log_returns_rest[cumulative_log_returns_rest.index >= '2006-01-01']

        # Plot cumulative log returns
        st.line_chart(cumulative_log_returns_rest, x_label="Year", y_label="Cumulative Log Return", height=455,
                      width=700, use_container_width=False)
        st.write("Note: Crypto is not in the graph because ...")




# streamlit run draft_web.py


