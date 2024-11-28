from utilities import *

config = {
    'limit_year': None,
    'data_frequency': "monthly",
    'rebalancing_frequency': "annual",
    'ANNUALIZATION_FACTOR': 12,
    'master_index': None,
    'global_tickers': None,
    'mode': 'gamma', # 'fast' or 'gamma' for frontier optimization
    'gamma_linspace': np.linspace(-0.5, 2, 3)} # 101
# print(np.linspace(-0.5, 2, int(250/2+1)))

settings.update_settings(**config)

spinner = Spinner("Starting...")
spinner.start()
spinner.message("Loading data...", "blue")

root = os.path.dirname(__file__)
equity_path = os.path.join(root, 'data', 'all_prices.csv')
cap_path = os.path.join(root, 'data', 'cap_data.csv')
rf_path = os.path.join(root, 'data', 'rf_rate.csv')

all_prices = pd.read_csv(equity_path, header=[0, 1], index_col=0, parse_dates=True)
all_caps = pd.read_csv(cap_path, index_col=0, parse_dates=True)
rf_rate = pd.read_csv(rf_path, index_col=0, parse_dates=True)

all_returns = all_prices.ffill().pct_change()
all_returns[all_prices.isna()] = None
all_returns.replace([np.inf, -np.inf, -1], np.nan, inplace=True)
all_returns.fillna(0, inplace=True)

masterIndex = all_returns.index
rf_rate = rf_rate.loc[masterIndex]
settings.update_settings(master_index=masterIndex)
settings.update_settings(global_tickers=list(all_returns.columns.get_level_values(1)))

indexIterator = iteration_depth()
spinner.message('Optimizing', 'yellow')

portfolio_keys = ['equity_amer', 'equity_em', 'equity_eur', 'equity_pac', 'metals', 'commodities', 'crypto', 'volatilities']
portfolio_returns = pd.DataFrame(index=masterIndex, columns=[*portfolio_keys, 'erc'])

portfolio_weights = []
visual_data = {}

if settings.mode == 'gamma':
    portfolio_gamma_collector = {}
    portfolio_gamma_returns = pd.DataFrame(None, index=masterIndex, columns=pd.MultiIndex.from_product([settings.gamma_linspace, portfolio_keys]))
    portfolio_erc_returns = pd.DataFrame(None, index=masterIndex, columns=pd.MultiIndex.from_product([settings.gamma_linspace, ['erc']]))

start_time = time.time()
for step in indexIterator:
    
    spinner.erase()
    spinner.message(f'Optimizing {step+1}/{len(indexIterator)}...', 'yellow')

    optimizationIndex = indexIterator[step]['optimizationIndex']
    evaluationIndex   = indexIterator[step]['evaluationIndex']

    sampleReturns     = all_returns.loc[optimizationIndex].sort_index(axis=1)
    sampleRf = rf_rate.loc[optimizationIndex].iloc[-1].values[0]
    evaluationReturns = all_returns.loc[evaluationIndex].sort_index(axis=1)

    minCap = 0
    maxCap = np.inf
    nullFilter = create_filter_mask1(sampleReturns[portfolio_keys[:-4]], all_caps, minCap, maxCap)

    sampleReturns = sampleReturns.drop(columns=nullFilter)
    evaluationReturns = evaluationReturns.drop(columns=nullFilter)

    # Equity and Commodities Portfolios
    # Availible modes: 'min_var', 'max_sharpe', 'erc'
    mode = ...
    equityPortfolioAMER   = Portfolio(sampleReturns[portfolio_keys[0]], 'max_sharpe', risk_free_rate=sampleRf)
    equityPortfolioEM     = Portfolio(sampleReturns[portfolio_keys[1]], 'max_sharpe', risk_free_rate=sampleRf)
    equityPortfolioEUR    = Portfolio(sampleReturns[portfolio_keys[2]], 'max_sharpe', risk_free_rate=sampleRf)
    equityPortfolioPAC    = Portfolio(sampleReturns[portfolio_keys[3]], 'max_sharpe', risk_free_rate=sampleRf)

    metalsPortfolio       = Portfolio(sampleReturns[portfolio_keys[4]], 'max_sharpe', risk_free_rate=sampleRf)
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

    if settings.mode == 'gamma':
        portfolios = [equityPortfolioAMER, equityPortfolioEM, equityPortfolioEUR, equityPortfolioPAC, 
                      metalsPortfolio, commoditiesPortfolio, cryptoPortfolio, volatilitiesPortfolio]
        # Portfolio returns for each gamma
        for gamma in settings.gamma_linspace:
            for i, portfolio in enumerate(portfolios):
                performance = portfolio.log_performance(evaluationReturns[portfolio_keys[i]])
                portfolio_gamma_collector[(gamma, portfolio_keys[i])] = performance.loc[slice(None), gamma].values

        collector = pd.DataFrame(portfolio_gamma_collector)
        collector.columns = pd.MultiIndex.from_tuples(collector.columns, names=["gamma", "portfolio"])
        collector.index = portfolio_returns.loc[evaluationIndex, slice(None)].index
        portfolio_gamma_returns.loc[evaluationIndex] = collector

        # ERC Portfolio for each gamma
        pseudo_frontier = pd.DataFrame(None, index=settings.gamma_linspace, columns=['expected_return', 'expected_variance', 'expected_sharpe', *portfolio_keys])

        for gamma in settings.gamma_linspace:
            samplePortfolio = portfolio_gamma_returns.loc[optimizationIndex, gamma]
            evaluationPortfolio = portfolio_gamma_returns.loc[evaluationIndex, gamma]

            ercPortfolio = Portfolio(samplePortfolio[portfolio_keys], 'erc', trust_markowitz=False, main=True, erc_gamma_mode=gamma)
            portfolio_erc_returns.loc[evaluationIndex, (gamma, 'erc')] = ercPortfolio.evaluate_performance(evaluationPortfolio[portfolio_keys]).values
            
            pseudo_frontier.loc[gamma, 'expected_return'] = ercPortfolio.expected_portfolio_return
            pseudo_frontier.loc[gamma, 'expected_variance'] = ercPortfolio.expected_portfolio_varcov
            sharpeRatio = ercPortfolio.expected_portfolio_return / np.sqrt(ercPortfolio.expected_portfolio_varcov) if ercPortfolio.expected_portfolio_varcov > 0 else 0
            pseudo_frontier.loc[gamma, 'expected_sharpe'] = sharpeRatio

            for i, asset in enumerate(portfolio_keys):
                pseudo_frontier.loc[gamma, asset] = ercPortfolio.optimal_weights[i]

        print(pseudo_frontier)
        # Adding ERC to the frontier data
        pseudo_portfolio = Pseudo(pseudo_frontier, portfolio_keys)
        local_tickers = [*settings.global_tickers, *portfolio_keys]
        for portfolio, portfolio_name in zip([*portfolios, pseudo_portfolio], [*portfolio_keys, 'erc']):
            tickers = portfolio.ticker
            frontier = portfolio.frontier

            expected_returns = frontier['expected_return'].values
            expected_variances = frontier['expected_variance'].values
            expected_sharpes = frontier['expected_sharpe'].values
            weights = frontier.loc[:, tickers].values
            
            for i, gamma in enumerate(settings.gamma_linspace):
                row_data = [expected_returns[i], expected_variances[i], expected_sharpes[i]]
                
                weight_row = [np.nan] * len(local_tickers)
                for j, asset in enumerate(tickers):
                    asset_index = local_tickers.index(asset)
                    weight_row[asset_index] = weights[i, j]
                
                row_data.extend(weight_row)
                visual_data[(f"{step + 2006}-01-01", gamma, portfolio_name)] = row_data

    # ERC Portfolio with recommended gamma
    samplePortfolio = portfolio_returns.loc[optimizationIndex]
    evaluationPortfolio = portfolio_returns.loc[evaluationIndex]
    ercPortfolio = Portfolio(samplePortfolio[portfolio_keys], 'erc', trust_markowitz=False, main=True, fast_erc=True)
    portfolio_returns.loc[evaluationIndex, 'erc'] = ercPortfolio.evaluate_performance(evaluationPortfolio[portfolio_keys]).values
                
    step_weights = []
    for portfolio, category in zip([*Portfolio.non_combined_portfolios, ercPortfolio], [*portfolio_keys, 'erc']):
        weights = portfolio.actual_weights
        weights.columns = pd.MultiIndex.from_product([[category], weights.columns])
        step_weights.append(weights)
    portfolio_weights.append(pd.concat(step_weights, axis=1))

    Portfolio.non_combined_portfolios = []

portfolio_weights = pd.concat(portfolio_weights, axis=0).reindex(columns=all_returns.columns.append(pd.MultiIndex.from_product([['erc'], portfolio_keys])))
portfolio_returns.to_csv(os.path.join(root, 'data', 'portfolio_returns.csv'))
portfolio_weights.to_csv(os.path.join(root, 'data', 'portfolio_weights.csv'))

if settings.mode == 'gamma':
    pd.set_option('future.no_silent_downcasting', True)
    limit_year = settings.limit_year if settings.limit_year else 2021
    base_path = os.path.join(root, "data")
    visual_base_filename = "efficient_frontiers_gamma" # remove gamma to generate old files
    return_base_filename = "portfolio_returns_gamma"

    # Returns for each gamma
    portfolio_gamma_returns = pd.concat([portfolio_gamma_returns, portfolio_erc_returns], axis=1)
    portfolio_gamma_returns = portfolio_gamma_returns.loc["2006":str(limit_year)]
    portfolio_gamma_returns = portfolio_gamma_returns.stack(level=0, future_stack=True)
    portfolio_gamma_returns.index.names = ['date', 'gamma']
    split_large_csv(portfolio_gamma_returns, base_path, return_base_filename, max_size_mb=40)

    index = pd.MultiIndex.from_tuples(visual_data.keys(), names=["date", "gamma", "portfolio"])
    columns = pd.MultiIndex.from_tuples(
        [("expected_return", ""), ("expected_variance", ""), ("expected_sharpe", "")] +
        [(asset, "") for asset in [*settings.global_tickers, *portfolio_keys]],
        names=["attribute", "detail"])
    
    visual_df = pd.DataFrame.from_dict(visual_data, orient="index", columns=columns)
    visual_df.index = index
    visual_df = visual_df.reset_index(level=["portfolio"])
    visual_df = visual_df.reset_index().set_index(["date", "gamma", "portfolio"])
    visual_df.columns = [col[0] for col in visual_df.columns]
    visual_df.dropna(how="all", inplace=True)
    split_large_csv(visual_df, base_path, visual_base_filename, max_size_mb=40)

spinner.erase()
spinner.message('Done!\n', 'green')
spinner.stop()

print((1 + portfolio_returns[portfolio_returns.index.year <= 2021]).cumprod().tail(1))
print(portfolio_evaluation(portfolio_returns, pd.Series(0, index=portfolio_returns.index))['SR'])
print(portfolio_evaluation(portfolio_returns['erc'], pd.Series(0, index=portfolio_returns.index)))
print(f"Optimization Runtime: {(time.time() - start_time):2f}s")

# # Old saving backup
# # portfolio_weights = pd.concat(portfolio_weights, axis=0).reindex(columns=all_returns.columns.append(pd.MultiIndex.from_product([['erc'], portfolio_keys])))
# # portfolio_returns.to_csv(os.path.join(root, 'data', 'portfolio_returns.csv'))
# # portfolio_weights.to_csv(os.path.join(root, 'data', 'portfolio_weights.csv'))

# # pd.set_option('future.no_silent_downcasting', True)
# # base_path = os.path.join(root, "data")
# # visual_base_filename = "efficient_frontiers"

# # index = pd.MultiIndex.from_tuples(visual_data.keys(), names=["year", "gamma", "portfolio"])
# # columns = pd.MultiIndex.from_tuples(
# #     [("metrics", "expected_return"), ("metrics", "expected_variance"), ("metrics", "expected_sharpe")] +
# #     [("weights", asset) for asset in [*settings.global_tickers, *portfolio_keys]],
# #     names=["category", "attribute"])
# # visual_df = pd.DataFrame.from_dict(visual_data, orient="index", columns=columns)
# # visual_df.index = index
# # split_large_csv(visual_df, base_path, visual_base_filename, max_size_mb=50)