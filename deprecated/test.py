from utilities import *
        
spinner = Spinner("Starting...")
spinner.start()
spinner.message("Loading data...", "blue")

config = {
    'limit_year': None,
    'data_frequency': "monthly",
    'rebalancing_frequency': "annual",
    'ANNUALIZATION_FACTOR': 12,
    'master_index': None,
    'mode': 'fast',}

settings.update_settings(**config)

root = os.path.dirname(__file__)
staticPath = os.path.join(root, 'data', 'Static.xlsx')
ritPath = os.path.join(root, 'data', 'DS_RI_T_USD_M.xlsx')
mvPath = os.path.join(root, 'data', 'DS_MV_USD_M.xlsx')
rfPath = os.path.join(root, 'data', 'Risk_Free_Rate.xlsx')
nePath = os.path.join(root, 'data', 'data_YF/master_df.csv')

staticData = pd.read_excel(staticPath, engine='openpyxl')

masterData = pd.read_excel(ritPath, usecols=lambda x: x != 'NAME', index_col=0, engine='openpyxl').transpose()
capData = pd.read_excel(mvPath, usecols=lambda x: x != 'NAME', index_col=0, engine='openpyxl').transpose()

masterData.index.rename('DATE', inplace=True)
capData.index = pd.to_datetime(capData.index, format='%Y-%m-%d')
capData.index.rename('DATE', inplace=True)

masterData = masterData[masterData.index.year > 2000]
capData = capData[capData.index.year > 2000] * 1e6

masterIndex = masterData.index
settings.update_settings(master_index=masterIndex)

equity_portfolios = {
    'equity_amer': ['AMER'],
    'equity_em': ['EM'],
    'equity_eur': ['EUR'],
    'equity_pac': ['PAC']}

non_equity_portfolios = {
    'metals': ['Gold', 'Silver', 'Platinum', 'Palladium', 'Copper'],
    'commodities': ['Corn', 'Crude_Oil', 'Lean_Hogs', 'Live_Cattle', 'Natural_Gas', 'Soybeans', 'Wheat'],
    'crypto': ['Bitcoin', 'Ethereum'],
    'volatilities': ['VIX', 'MOVE_bond_market_volatility', 'VVIX_VIX_of_VIX', 'VXO-S&P_100_volatility', 'Nasdaq_VXN', 'Russell_2000_RVX']}

df_dict = {}
mv_dict = {}
for region in ['AMER', 'EM', 'EUR', 'PAC']:
    filter = staticData['ISIN'][staticData['Region'] == region]
    df_dict[region] = masterData[filter].pct_change()
    mv_dict[region] = capData[filter]
equity_returns = pd.concat(df_dict.values(), keys=df_dict.keys(), axis=1)
market_values = pd.concat(mv_dict.values(), keys=mv_dict.keys(), axis=1)



root = os.path.dirname(__file__)
equity_path = os.path.join(root, 'data', 'all_prices.csv')

non_equities = pd.read_csv(equity_path, header=[0, 1], index_col=0, parse_dates=True)
non_equities = non_equities.reindex(masterIndex).pct_change()
# for portoflio_name, tickers in non_equity_portfolios:

    

commodities = {
    "Gold": "GC=F",
    "Silver": "SI=F",}

metal_returns = pd.DataFrame(index=masterIndex)
for name, ticker in commodities.items():
    data = yf.download(ticker, start=min(masterIndex), end=max(masterIndex), interval='1d')
    data_filled = data['Close'].reindex(masterIndex).ffill()
    metal_returns[name] = data_filled.pct_change()

for df in [equity_returns, metal_returns, non_equities]:
    df.replace([np.inf, -np.inf, -1], np.nan, inplace=True)
    df.fillna(0, inplace=True)

portfolio_keys = ['equity_amer', 'equity_em', 'equity_pac', 'equity_eur', 'metals', 'commodities', 'crypto', 'volatilities']
portfolio_returns = pd.DataFrame(index=masterIndex, columns=[*portfolio_keys, 'ERC'])
portfolio_returns[:] = 0

indexIterator = iteration_depth()
spinner.message('Optimizing', 'yellow')

start_time = time.time()
for step in indexIterator:
    
    spinner.erase()
    spinner.message(f'Optimizing {step+1}/{len(indexIterator)}...', 'yellow')

    optimizationIndex = indexIterator[step]['optimizationIndex']
    evaluationIndex = indexIterator[step]['evaluationIndex']

    sampleEquity = equity_returns.loc[optimizationIndex]
    sampleMarketValues = market_values.loc[optimizationIndex]
    sampleMetals = metal_returns.loc[optimizationIndex]
    sampleNonEquities = non_equities.loc[optimizationIndex]
    evaluationEquity = equity_returns.loc[evaluationIndex]
    evaluationMetals = metal_returns.loc[evaluationIndex]
    evaluationNonEquities = non_equities.loc[evaluationIndex]

    minMarketCapThreshold = 0
    maxMarketCapThreshold = 100e9
    nullFilter = create_filter_mask(sampleEquity, sampleMarketValues, minMarketCapThreshold, maxMarketCapThreshold)

    a = sampleEquity.shape[1]
    sampleEquity = sampleEquity.drop(columns=nullFilter)
    evaluationEquity = evaluationEquity.drop(columns=nullFilter)
    print(f'Dropped {a - sampleEquity.shape[1]} equities')

    # Equity and Commodities Portfolios
    equityPortfolioAMER = Portfolio(sampleEquity['AMER'], 'min_var')
    equityPortfolioEM = Portfolio(sampleEquity['EM'], 'min_var')
    equityPortfolioEUR = Portfolio(sampleEquity['EUR'], 'min_var')
    equityPortfolioPAC = Portfolio(sampleEquity['PAC'], 'min_var')

    metalsPortfolio = Portfolio(sampleNonEquities['metals'], 'min_var')
    commoditiesPortfolio = Portfolio(sampleNonEquities['commodities'], 'min_var')
    cryptoPortfolio = Portfolio(sampleNonEquities['crypto'], 'min_var')
    volatilitiesPortfolio = Portfolio(sampleNonEquities['volatilities'], 'min_var')

    portfolio_returns.loc[evaluationIndex, 'equity_amer'] = equityPortfolioAMER.evaluate_performance(evaluationEquity['AMER']).values
    portfolio_returns.loc[evaluationIndex, 'equity_em'] = equityPortfolioEM.evaluate_performance(evaluationEquity['EM']).values
    portfolio_returns.loc[evaluationIndex, 'equity_eur'] = equityPortfolioEUR.evaluate_performance(evaluationEquity['EUR']).values
    portfolio_returns.loc[evaluationIndex, 'equity_pac'] = equityPortfolioPAC.evaluate_performance(evaluationEquity['PAC']).values
    
    portfolio_returns.loc[evaluationIndex, 'metals'] = metalsPortfolio.evaluate_performance(evaluationNonEquities['metals']).values
    portfolio_returns.loc[evaluationIndex, 'commodities'] = commoditiesPortfolio.evaluate_performance(evaluationNonEquities['commodities']).values
    portfolio_returns.loc[evaluationIndex, 'crypto'] = cryptoPortfolio.evaluate_performance(evaluationNonEquities['crypto']).values
    portfolio_returns.loc[evaluationIndex, 'volatilities'] = volatilitiesPortfolio.evaluate_performance(evaluationNonEquities['volatilities']).values

    # ERC Portfolio
    samplePortfolio = portfolio_returns.loc[optimizationIndex]
    evaluationPortfolio = portfolio_returns.loc[evaluationIndex]

    ercPortfolio = Portfolio(samplePortfolio[portfolio_keys], 'erc', main = True, trust_markowitz=False)

    portfolio_returns.loc[evaluationIndex, 'ERC'] = ercPortfolio.evaluate_performance(evaluationPortfolio[portfolio_keys]).values


    Portfolio.non_combined_portfolios = []
    

spinner.erase()
spinner.message('Done!\n', 'green')
spinner.stop()

print((1 + portfolio_returns[portfolio_returns.index.year < 2022]).cumprod().tail(1))
# sharpe ratio of portfolios
print(portfolio_evaluation(portfolio_returns, pd.Series(0, index=portfolio_returns.index))['SR'])
# print(portfolio_evaluation(portfolio_returns['metals'], pd.Series(0, index=portfolio_returns.index)))
print(portfolio_evaluation(portfolio_returns['ERC'], pd.Series(0, index=portfolio_returns.index)))
print(f"Optimization Runtime: {(time.time() - start_time):2f}s")

# Pre change = post change - minvar
#             equity_amer equity_em equity_pac equity_eur    metals       ERC
# DATE                                                                      
# 2021-12-31    5.038872  8.959687   3.859669   2.162318  3.694905  4.797688
# equity_amer    0.782514
# equity_em      0.930665
# equity_pac     0.663112
# equity_eur     0.365398
# metals         0.487281
# ERC            0.932358
# dtype: object
# {'mu': 0.07762197680093252, 'std': 0.08325340507787439, 'SR': 0.9323579825753159, 'min': -0.15013698857945015, 'max': 0.07431939228752919}
# Optimization Runtime: 11.776785s