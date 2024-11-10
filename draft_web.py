import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
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

# Help for make the web clean
# --------------------------------------------------------------------------------------
# Name of portfolio types
list_type_portfolio = ['equity_amer', 'equity_em', 'equity_eur', 'equity_pac',
                       'metals','commodities', 'crypto', 'volatilities']

list_clean_name = ['Metals','Commodities', 'Crypto', 'Volatilities',
                   'North American Equities','Emerging Markets Equities','European Equities', 'Asia-Pacific Equities']

# decomposition portfolio
list_commodities = ["Lean_Hogs","Crude_Oil","Live_Cattle","Soybeans","Wheat","Corn","Natural_Gas"]
list_crypto = ["Bitcoin","Ethereum"]
list_metals = ["Gold","Platinum","Palladium","Silver","Copper"]
list_volatilities = ["Russell_2000_RVX","VVIX_VIX_of_VIX","MOVE_bond_market_volatility",
                     "VXO-S&P_100_volatility","Nasdaq_VXN","VIX"]

# Data Exploration
df_commodities = master_df[list_commodities].pct_change()
df_crypto = master_df[list_crypto].pct_change()
df_metals = master_df[list_metals].pct_change()
df_volatilities = master_df[list_volatilities].pct_change()
corr_matrix = master_df.corr()

# list ISIN of equities
list_data_equity_path = os.path.join(root, 'data', 'list_equity')
list_data_equity_amer = pd.read_csv(os.path.join(list_data_equity_path, "equity_amer.csv"))
list_data_equity_em = pd.read_csv(os.path.join(list_data_equity_path, "equity_em.csv"))
list_data_equity_eur = pd.read_csv(os.path.join(list_data_equity_path, "equity_eur.csv"))
list_data_equity_pac = pd.read_csv(os.path.join(list_data_equity_path, "equity_pac.csv"))


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

    if selection in ["Commodities","Metals","Crypto","Volatilities"]:

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


        st.subheader("Expected returns and volatilities", divider="gray")
        st.write("Expected returns")
        st.bar_chart(data_use.mean())
        st.write("Expected volatilities")
        st.bar_chart(data_use.std())
        st.subheader("Heatmap", divider="gray")
        st.write("")
        plot = sns.heatmap(correl_matrix, annot=True, cmap="coolwarm")
        st.pyplot(plot.get_figure())

    if selection in ['North American Equities','Emerging Markets Equities','European Equities', 'Asia-Pacific Equities']:
        st.info("The data exploration is different for equities because of the large numbers of securities")

        if selection == "North American Equities":
            mean = str(round(float(portfolio_returns["equity_amer"].mean() * 100),2))
            vol = str(round(float(portfolio_returns["equity_amer"].std() * 100),2))
            nb_eq = int(list_data_equity_amer.count())
            list_data_equity_amer.rename(columns={"equity_amer":"ISIN"},inplace=True)
            list_isin = list_data_equity_amer

        if selection == "Emerging Markets Equities":
            mean = str(round(float(portfolio_returns["equity_em"].mean() * 100),2))
            vol = str(round(float(portfolio_returns["equity_em"].std() * 100),2))
            nb_eq = int(list_data_equity_em.count())
            list_isin = list_data_equity_em

        if selection == "European Equities":
            mean = str(round(float(portfolio_returns["equity_eur"].mean() * 100),2))
            vol = str(round(float(portfolio_returns["equity_eur"].std() * 100),2))
            nb_eq = int(list_data_equity_eur.count())
            list_isin = list_data_equity_eur

        if selection == "Asia-Pacific Equities":
            mean = str(round(float(portfolio_returns["equity_pac"].mean() * 100),2))
            vol = str(round(float(portfolio_returns["equity_pac"].std() * 100),2))
            nb_eq = int(list_data_equity_pac.count())
            list_isin = list_data_equity_pac

        st.subheader("Expected return and volatility", divider="gray")
        st.markdown("**Expected** **return**: " + mean + "%")
        st.markdown("**Expected** **volatility**: " + vol + "%")
        st.write("")
        st.subheader("Additional informations", divider="gray")
        st.markdown("**Number of equities**: " + str(nb_eq))
        st.write("")
        st.markdown("**Equities composition**:")
        st.write(list_isin)

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

#***********************************************************************************************************
# Performance
#***********************************************************************************************************
if choice == "Performance":

    st.title("Performance")
    st.subheader("Cumulative return", divider="gray")

    choice = st.toggle("Our strategy",value=True)

    if choice == True:
        cumulative_returns = (1 + portfolio_returns["erc"]).cumprod()
        cumulative_returns = cumulative_returns[cumulative_returns.index >= '2006-01-01']
        st.line_chart(cumulative_returns,color="#1ABC9C" , x_label="Year", y_label="cumulative return (in %)",height=400,width=700,use_container_width=False)

    if choice == False:
        list_ = ['equity_amer', 'equity_em', 'equity_eur', 'equity_pac','metals','commodities', 'volatilities','erc']

        cumulative_returns_rest = (1 + portfolio_returns[list_]).cumprod()
        cumulative_returns_rest = cumulative_returns_rest[cumulative_returns_rest.index >= '2006-01-01']
        st.line_chart(cumulative_returns_rest, x_label="Year", y_label="cumulative return (in %)",height=455,width=700,use_container_width=False)
        st.write("Note: Crypto is not in the graph because ...")