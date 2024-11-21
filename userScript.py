from utilities import *

config = {
    'limit_year': None,
    'data_frequency': "monthly",
    'rebalancing_frequency': "annual",
    'ANNUALIZATION_FACTOR': 12,
    'master_index': None,
    'global_tickers': None,
    'mode': 'gamma', # 'fast' or 'gamma' for frontier optimization
    'gamma_linspace': np.linspace(-0.5, 1.5, 101)} # 101

settings.update_settings(**config)

spinner = Spinner("Starting...")
spinner.start()
spinner.message("Loading data...", "blue")

root = os.path.dirname(__file__)
equity_path = os.path.join(root, 'data', 'all_prices.csv')
cap_path = os.path.join(root, 'data', 'cap_data.csv')

all_prices = pd.read_csv(equity_path, header=[0, 1], index_col=0, parse_dates=True)
all_caps = pd.read_csv(cap_path, index_col=0, parse_dates=True)

all_returns = all_prices.ffill().pct_change()
all_returns[all_prices.isna()] = None
all_returns.replace([np.inf, -np.inf, -1], np.nan, inplace=True)
all_returns.fillna(0, inplace=True)

masterIndex = all_returns.index
settings.update_settings(master_index=masterIndex)
settings.update_settings(global_tickers=list(all_returns.columns.get_level_values(1)))

indexIterator = iteration_depth()
spinner.message('Optimizing', 'yellow')

portfolio_keys = ['equity_amer', 'equity_em', 'equity_eur', 'equity_pac', 'metals', 'commodities', 'crypto', 'volatilities']
portfolio_returns = pd.DataFrame(index=masterIndex, columns=[*portfolio_keys, 'erc'])

portfolio_weights = []
visual_data = {}

start_time = time.time()
for step in indexIterator:
    
    spinner.erase()
    spinner.message(f'Optimizing {step+1}/{len(indexIterator)}...', 'yellow')

    optimizationIndex = indexIterator[step]['optimizationIndex']
    evaluationIndex   = indexIterator[step]['evaluationIndex']

    sampleReturns     = all_returns.loc[optimizationIndex].sort_index(axis=1)
    evaluationReturns = all_returns.loc[evaluationIndex].sort_index(axis=1) 

    minCap = 0
    maxCap = np.inf
    nullFilter = create_filter_mask1(sampleReturns[portfolio_keys[:-4]], all_caps, minCap, maxCap)

    sampleReturns = sampleReturns.drop(columns=nullFilter)
    evaluationReturns = evaluationReturns.drop(columns=nullFilter)

    # Equity and Commodities Portfolios
    # Availible modes: 'min_var', 'max_sharpe', 'erc'
    mode = ...
    equityPortfolioAMER   = Portfolio(sampleReturns[portfolio_keys[0]], 'max_sharpe')
    equityPortfolioEM     = Portfolio(sampleReturns[portfolio_keys[1]], 'max_sharpe')
    equityPortfolioEUR    = Portfolio(sampleReturns[portfolio_keys[2]], 'max_sharpe')
    equityPortfolioPAC    = Portfolio(sampleReturns[portfolio_keys[3]], 'max_sharpe')

    metalsPortfolio       = Portfolio(sampleReturns[portfolio_keys[4]], 'max_sharpe')
    commoditiesPortfolio  = Portfolio(sampleReturns[portfolio_keys[5]], 'min_var')
    cryptoPortfolio       = Portfolio(sampleReturns[portfolio_keys[6]], 'min_var')
    volatilitiesPortfolio = Portfolio(sampleReturns[portfolio_keys[7]], 'erc')

    portfolio_returns.loc[evaluationIndex, portfolio_keys[0]] =   equityPortfolioAMER.evaluate_performance(evaluationReturns[portfolio_keys[0]]).values
    portfolio_returns.loc[evaluationIndex, portfolio_keys[1]] =     equityPortfolioEM.evaluate_performance(evaluationReturns[portfolio_keys[1]]).values
    portfolio_returns.loc[evaluationIndex, portfolio_keys[2]] =    equityPortfolioEUR.evaluate_performance(evaluationReturns[portfolio_keys[2]]).values
    portfolio_returns.loc[evaluationIndex, portfolio_keys[3]] =    equityPortfolioPAC.evaluate_performance(evaluationReturns[portfolio_keys[3]]).values

    portfolio_returns.loc[evaluationIndex, portfolio_keys[4]] =       metalsPortfolio.evaluate_performance(evaluationReturns[portfolio_keys[4]]).values
    portfolio_returns.loc[evaluationIndex, portfolio_keys[5]] =  commoditiesPortfolio.evaluate_performance(evaluationReturns[portfolio_keys[5]]).values
    portfolio_returns.loc[evaluationIndex, portfolio_keys[6]] =       cryptoPortfolio.evaluate_performance(evaluationReturns[portfolio_keys[6]]).values
    portfolio_returns.loc[evaluationIndex, portfolio_keys[7]] = volatilitiesPortfolio.evaluate_performance(evaluationReturns[portfolio_keys[7]]).values


    # ERC Portfolio
    samplePortfolio = portfolio_returns.loc[optimizationIndex]
    evaluationPortfolio = portfolio_returns.loc[evaluationIndex]
    # print(samplePortfolio[portfolio_keys])
    # print(evaluationPortfolio[portfolio_keys])

    ercPortfolio = Portfolio(samplePortfolio[portfolio_keys], 'erc', trust_markowitz=False, main=True)

    portfolio_returns.loc[evaluationIndex, 'erc'] = ercPortfolio.evaluate_performance(evaluationPortfolio[portfolio_keys]).values


    step_weights = []
    for portfolio, category in zip([*Portfolio.non_combined_portfolios, ercPortfolio], [*portfolio_keys, 'erc']):
        weights = portfolio.actual_weights
        weights.columns = pd.MultiIndex.from_product([[category], weights.columns])
        step_weights.append(weights)
    portfolio_weights.append(pd.concat(step_weights, axis=1))

    if settings.mode == 'gamma':
        portfolios = [equityPortfolioAMER, equityPortfolioEM, equityPortfolioEUR, equityPortfolioPAC, metalsPortfolio, commoditiesPortfolio, cryptoPortfolio, volatilitiesPortfolio]
        portfolio_names = portfolio_keys
        for portfolio, portfolio_name in zip(portfolios, portfolio_names):
            tickers = portfolio.ticker
            frontier = portfolio.frontier

            expected_returns = frontier['expected_return'].values
            expected_variances = frontier['expected_variance'].values
            expected_sharpes = frontier['expected_sharpe'].values
            weights = frontier.loc[:, tickers].values
            
            for i, gamma in enumerate(settings.gamma_linspace):
                row_data = [expected_returns[i], expected_variances[i], expected_sharpes[i]]
                
                weight_row = [np.nan] * len(settings.global_tickers)
                for j, asset in enumerate(tickers):
                    asset_index = settings.global_tickers.index(asset)
                    weight_row[asset_index] = weights[i, j]
                
                row_data.extend(weight_row)
                visual_data[(step, gamma, portfolio_name)] = row_data

    Portfolio.non_combined_portfolios = []

portfolio_weights = pd.concat(portfolio_weights, axis=0).reindex(columns=all_returns.columns.append(pd.MultiIndex.from_product([['erc'], portfolio_keys])))
portfolio_returns.to_csv(os.path.join(root, 'data', 'portfolio_returns.csv'))
portfolio_weights.to_csv(os.path.join(root, 'data', 'portfolio_weights.csv'))

if settings.mode == 'gamma':
    index = pd.MultiIndex.from_tuples(visual_data.keys(), names=["year", "gamma", "portfolio"])
    columns = pd.MultiIndex.from_tuples(
        [("metrics", "expected_return"), ("metrics", "expected_variance"), ("metrics", "expected_sharpe")] +
        [("weights", asset) for asset in settings.global_tickers],
        names=["category", "attribute"])

    visual_df = pd.DataFrame.from_dict(visual_data, orient="index", columns=columns)
    visual_df.index = index
    base_path = os.path.join(root, "data")
    base_filename = "efficient_frontiers"
    split_large_csv(visual_df, base_path, base_filename, max_size_mb=50)

spinner.erase()
spinner.message('Done!\n', 'green')
spinner.stop()

print((1 + portfolio_returns[portfolio_returns.index.year <= 2021]).cumprod().tail(1))
print(portfolio_evaluation(portfolio_returns, pd.Series(0, index=portfolio_returns.index))['SR'])
print(portfolio_evaluation(portfolio_returns['erc'], pd.Series(0, index=portfolio_returns.index)))
print(f"Optimization Runtime: {(time.time() - start_time):2f}s")