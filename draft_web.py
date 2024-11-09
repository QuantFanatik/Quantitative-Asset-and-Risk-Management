import streamlit as st
import numpy as np
import pandas as pd
import os
import seaborn as sns

# Data Source 1
# ---------------------------------------------------------------------------------------
root = os.path.dirname(__file__)
returns_path = os.path.join(root, 'data', 'portfolio_returns.csv')
weights_path = os.path.join(root, 'data', 'portfolio_weights.csv')

portfolio_returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
portfolio_weights = pd.read_csv(weights_path, index_col=0, parse_dates=True, header=[0, 1])

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
# ---------------------------------------------------------------------------------------

# -----------------------------
# Principal bar
# -----------------------------
with st.sidebar:
    st.title("Portfolio Optimization")
    choice = st.radio("Steps", ["Data Exploration", "Equal Risk Contributions", "Optimal portfolio", "Performance"])
    st.info("This tool uses equal risk contributions method to select optimal weights for each "
            "type of asset classes. Then, we use Markowitz optimization to choose an optimal portfolio.")

# -----------------------------
# Data Exploration
# -----------------------------
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

        st.write("")
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

# -----------------------------
# Equal Risk Contributions
# -----------------------------
    if choice == "Data Exploration":
        pass