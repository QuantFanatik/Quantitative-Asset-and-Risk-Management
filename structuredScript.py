import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import Bounds, LinearConstraint, minimize
import scipy.sparse.linalg as sparla
import cvxpy as cp
import os
import yfinance as yf
import itertools, sys
import sys
import threading
import time
from itertools import cycle

global ANNUALIZATION_FACTOR
ANNUALIZATION_FACTOR = 12

class Spinner:
    def __init__(self, message="Processing...", color="white"):
        self.spinner = cycle(['|', '/', '-', '\\'])
        self.stop_running = threading.Event()
        self.message_text = message
        self.lock = threading.Lock()  # To prevent conflicts with message updates
        self.color_code = {
            "red": "\033[31m",
            "green": "\033[32m",
            "yellow": "\033[33m",
            "blue": "\033[34m",
            "white": "\033[37m",
            "reset": "\033[0m"
        }
        self.current_color = color 

    def start(self):
        def run_spinner():
            sys.stdout.write(self.message_text + " ")
            while not self.stop_running.is_set():
                with self.lock:
                    colored_symbol = self.color_code.get(self.current_color, self.color_code["white"]) + next(self.spinner) + self.color_code["reset"]
                    sys.stdout.write(colored_symbol)  
                    sys.stdout.flush()
                    sys.stdout.write('\b')
                time.sleep(0.1)

        self.thread = threading.Thread(target=run_spinner)
        self.thread.start()

    def stop(self):
        self.stop_running.set()
        self.thread.join()

    def message(self, new_message, color="white"):
        """Update the status message and color while the spinner is running."""
        with self.lock:
            sys.stdout.write('\b \b')  
            sys.stdout.flush()
            self.current_color = color
            colored_message = self.color_code.get(color, self.color_code["white"]) + new_message + self.color_code["reset"]
            sys.stdout.write('\r' + colored_message + " ")
            sys.stdout.flush()
            time.sleep(0.1)
            self.message_text = new_message

    def erase(self):
        """Erase the current message from the terminal."""
        with self.lock:
            sys.stdout.write('\r')
            sys.stdout.write(' ' * (len(self.message_text) + 2))
            sys.stdout.write('\r')
            sys.stdout.flush()
            self.message_text = ""

def excel_loader(path):
    data = pd.read_excel(path, usecols=lambda x: x != 'NAME', index_col=0).transpose()
    data.index = pd.to_datetime(data.index, format='%Y')
    data.index = data.index + pd.offsets.YearEnd()
    data.index.rename('DATE', inplace=True)
    data = data[data.index.year > 2004]
    nan_columns = data.iloc[0].loc[data.iloc[0].isna()].index
    data.loc['2005-12-31', nan_columns] = data.loc['2006-12-31', nan_columns]
    data.interpolate(method='linear', axis=0, inplace=True)

    return data

def annualized_mean(sample_mean: float) -> float:
    return (1 + sample_mean) ** ANNUALIZATION_FACTOR - 1

def annualized_volatility(sample_std: float) -> float:
    return sample_std * np.sqrt(ANNUALIZATION_FACTOR)

def sharpe_ratio(mean: float, volatility: float) -> float:
    return mean / volatility

def portfolio_evaluation(monthlyReturns: pd.Series | np.ndarray, monthlyRFrate: pd.Series) -> dict:

    '''
    Evaluates the performance of a portfolio given its monthly returns. 
    It calculates and returns a dictionary containing the annualized mean return,
    annualized volatility, Sharpe ratio, minimum return, and maximum return of the portfolio.
    monthlyRFrate must be indexed by and be of same length as the sample of monthly returns 
    that is being evaluated.
    '''

    mean = monthlyReturns.mean()
    volatility = monthlyReturns.std()
    annualizedMean = annualized_mean(mean)
    annualizedVolatility = annualized_volatility(volatility)
    monthlyExcessReturn = monthlyReturns.sub(monthlyRFrate, axis=0)
    meanExcessReturn = monthlyExcessReturn.mean()
    annualizedExcessReturn = annualized_mean(meanExcessReturn)
    sharpeRatio = sharpe_ratio(annualizedExcessReturn, annualizedVolatility)
    minimum = monthlyReturns.min()
    maximum = monthlyReturns.max()

    portfolio_performance = {
        'mu': annualizedMean,
        'std': annualizedVolatility,
        'SR': sharpeRatio,
        'min': minimum,
        'max': maximum
    }

    return portfolio_performance

def create_filter_mask(sampleData):
    
    highestYearEnd = sampleData.index.max()
    highestYearStart = pd.to_datetime(f'{highestYearEnd.year}-01-31')
    
    # Zero December Returns
    decemberData = sampleData.loc[[highestYearEnd]]
    decemberFilter = decemberData.columns[decemberData.iloc[0] == np.inf] # deactivated

    # December price below threshold
    yearEndPrices = masterData.loc[highestYearEnd]
    priceFilter = yearEndPrices[yearEndPrices < -np.inf].index # activated

    # High return filter
    returnFilterHigh = sampleData.columns[sampleData.max() >= np.inf] # deactivated
    returnFilterLow = sampleData.columns[sampleData.min() <= -np.inf] # deactivated
    returnFilter = returnFilterHigh.union(returnFilterLow)
    
    # Frequent Zero Returns
    yearlyData = sampleData.loc[highestYearStart:highestYearEnd]
    monthsWithZeroReturns = (yearlyData == 0).sum(axis=0)
    frequentZerosFilter = monthsWithZeroReturns[monthsWithZeroReturns >= 12].index # activated

    return decemberFilter.union(frequentZerosFilter).union(priceFilter).union(returnFilter)

class Portfolio():
    valid_types = ('markowitz', 'erc', 'max_sharpe', 'min_var')
    non_combined_portfolios = []

    def __init__(self, returns: pd.DataFrame | pd.Series, type: str='markowitz', names: list[str]=None, trust_markowitz: bool=False, resample: bool=False):
        assert type.lower() in self.valid_types, f"Invalid type: {type}. Valid types are: {self.valid_types}"
        #TODO: Attention! ERC portfolios use sample returns, not ex-ante expectations.
        self.trust_markowitz = trust_markowitz
        self.resample = resample
        self.type = type.lower()
        self.ticker = returns.columns
        self.returns = returns
        self.expected_returns = self.get_expected_returns()
        self.expected_covariance = self.get_expected_covariance()
        self.dim = len(self.expected_returns)
        self.len = len(self.returns)

        self.optimal_weights = self.get_optimize()
        self.expected_portfolio_return = self.get_expected_portfolio_return()
        self.expected_portfolio_varcov = self.get_expected_portfolio_varcov()

        if self.type != 'erc':
            Portfolio.non_combined_portfolios.append(self)

    def get_expected_returns(self) -> pd.DataFrame | pd.Series:
        #TODO: Attention! If extending beyond ERC, if statement must be updated.
        if self.trust_markowitz and self.type == 'erc':
            internal_expectations = np.array([portfolio.expected_portfolio_return for portfolio in Portfolio.non_combined_portfolios])
            return pd.Series(internal_expectations, index=self.returns.columns)
        return self.returns.mean(axis=0)
    
    def get_expected_covariance(self) -> pd.DataFrame | pd.Series:
        if self.trust_markowitz and self.type == 'erc':
            internal_expectations = np.array([portfolio.expected_portfolio_varcov for portfolio in Portfolio.non_combined_portfolios])
            sample_correlations = self.returns.corr().fillna(0)
            varcov_matrix = np.outer(internal_expectations, internal_expectations) * sample_correlations
            return pd.DataFrame(varcov_matrix, index=self.returns.columns, columns=self.returns.columns)
        return self.returns.cov(ddof=0)
    
    def get_expected_portfolio_return(self) -> float:
        return np.dot(self.expected_returns, self.optimal_weights)
    
    def get_expected_portfolio_varcov(self) -> float:
        return self.optimal_weights.T @ self.expected_covariance @ self.optimal_weights

    def _select_method(self):
        #TODO: Separate min_var and markowitz
        if self.type == 'markowitz' or self.type == 'min_var':
            return self._fit_markowitz
        if self.type == 'max_sharpe':
            return self._fit_max_sharpe
        if self.type == 'erc':
            return self._fit_erc

    def get_optimize(self) -> np.ndarray:
        "Returns a numpy array of optimal weights"
        if self.resample:
            return self._resample()
        else:
            return self._select_method()()

    def _resample(self) -> np.ndarray:
        N_SUMULATIONS = 10 # 500

        method = self._select_method()
        original_moments = (self.expected_returns.copy(), self.expected_covariance.copy())
        simulated_weights = []

        for i in range(N_SUMULATIONS):
            np.random.seed(i)
            simulated_returns = np.random.multivariate_normal(self.expected_returns, self.expected_covariance, self.len)
            self.expected_returns = self._pandify(np.mean(simulated_returns, axis=0)) * ANNUALIZATION_FACTOR
            self.expected_covariance = self._pandify(np.cov(simulated_returns.T, ddof=0))
            self.optimal_weights = method()
            simulated_weights.append(self.optimal_weights)
        
        self.expected_returns, self.expected_covariance = original_moments
        combined_simulation_data = np.stack(simulated_weights, axis=0)
        # return pd.Series(combined_simulation_data.mean(axis=0), index=self.ticker)
        return combined_simulation_data.mean(axis=0)
    
    def _pandify(self, array: np.ndarray) -> pd.Series | pd.DataFrame:
        if array.ndim == 1:
            return pd.Series(array, index=self.ticker)
        else:
            return pd.DataFrame(array, index=self.ticker, columns=self.ticker)
        
    def _is_psd(self):
        """Check if a covariance matrix is PSD."""
        eigenvalues = np.linalg.eigvals(self.expected_covariance)
        return np.all(eigenvalues >= -1e-16)
    
    def _check_arpack_stability(self, tol=1e-16) -> bool:
        try:
            sparla.eigsh(self.expected_covariance.to_numpy(), k=1, which='SA', tol=tol)
            return True
        except sparla.ArpackNoConvergence:
            return False
    
    def _fit_markowitz(self) -> np.ndarray:        
        weights = cp.Variable(self.dim)
        portfolio_variance = cp.quad_form(weights, cp.psd_wrap(self.expected_covariance))
        objective = cp.Minimize(portfolio_variance)
        constraints = [cp.sum(weights) == 1, 
                    weights >= 0]
        problem = cp.Problem(objective, constraints)
        problem.solve(warm_start=True)
        if weights.value is None:
            result = self._fit_markowitz_robust()
        else:
            result = weights.value
        return result
    
    def _fit_markowitz_robust(self) -> np.ndarray:
        print("Covariance matrix non PSD. Attempting robust optimization.")
        Sigma = self.expected_covariance
        kwargs = {'fun': lambda x: np.dot(x, np.dot(Sigma, x)),
                  'jac': lambda x: 2 * np.dot(Sigma, x),
                  'x0': np.ones(self.dim) / self.dim,
                  'constraints': LinearConstraint(np.ones(self.dim), 1, 1),
                  'bounds': Bounds(0, 1),
                  'method': 'SLSQP',
                  'tol': 1e-10} #'tol': 1e-16
        return minimize(**kwargs).x
    
    def _fit_max_sharpe(self) -> np.ndarray:
        if self.expected_returns.isna().all().all() or (self.expected_returns == 0).all().all():
            print("No data available for evaluation.")
            return np.zeros(self.dim)
        
        proxy_weights = cp.Variable(self.dim)
        objective = cp.Minimize(cp.quad_form(proxy_weights, cp.psd_wrap(self.expected_covariance)))
        constraints = [proxy_weights @ self.expected_returns == 1, 
                       proxy_weights >= 0]
        problem = cp.Problem(objective, constraints)
        problem.solve(warm_start=True)

        if proxy_weights.value is None:
            result = self._fit_max_sharpe_robust()
        else:
            result = proxy_weights.value / np.sum(proxy_weights.value)
        return result
    
    def _fit_max_sharpe_robust(self) -> np.ndarray:
        print("Sharpe Ratio optimization failed to find a solution. Attempting robust optimization.")
        mu = self.expected_returns
        Sigma = self.expected_covariance
        kwargs = {'fun': lambda x: -np.dot(mu, x) / np.sqrt(np.dot(x, np.dot(Sigma, x))),
                  'jac': lambda x: -mu / np.sqrt(np.dot(x, np.dot(Sigma, x))) + np.dot(np.dot(mu, x), np.dot(x, Sigma)) / np.sqrt(np.dot(x, np.dot(Sigma, x))**3),
                  'x0': np.ones(self.dim) / self.dim,
                  'constraints': LinearConstraint(np.ones(self.dim), 1, 1),
                  'bounds': Bounds(0, 1),
                  'method': 'SLSQP',
                  'tol': 1e-6} #'tol': 1e-16
        return minimize(**kwargs).x
    
    def _fit_erc(self):
        weights = cp.Variable(self.dim)
        objective = cp.Minimize(0.5 * cp.quad_form(weights, self.expected_covariance))

        log_constraint_bound = -self.dim * np.log(self.dim) - 2  # -2 does not matter after rescaling
        log_constraint = cp.sum(cp.log(weights)) >= log_constraint_bound
        constraints = [weights >= 0, weights <= 1, log_constraint]

        problem = cp.Problem(objective, constraints)
        # problem.solve(solver=cp.SCS, eps=1e-12) # Results in e-27 precision   
        problem.solve(warm_start=True) # Results in e-27 precision    

        if weights.value is None:
            result = self._fit_erc_robust()
        else:
            result = weights.value / np.sum(weights.value)
        return result

    def _fit_erc_robust(self) -> np.ndarray:
        print("ERC optimization failed to find a solution. Attempting robust optimization.")
        def _ERC(x, cov_matrix):
            volatility = np.sqrt(x.T @ cov_matrix @ x)
            abs_risk_contribution = x * (cov_matrix @ x) / volatility
            mean = np.mean(abs_risk_contribution)
            return np.sum((abs_risk_contribution - mean)**2)
        
        bounds = Bounds(0, 1)
        lc = LinearConstraint(np.ones(self.dim), 1, 1)
        settings = {'tol': 1e-16, 'method': 'SLSQP'} # This tolerance is required to match cvxpy results
        res = minimize(_ERC, np.full(self.dim, 1/self.dim), args=(self.expected_covariance), constraints=[lc], bounds=bounds, **settings)
        return res.x
    
    def evaluate_performance(self, evaluationData: pd.DataFrame | pd.Series) -> pd.Series:
        # Returns Adjusted for Return-Shifted Weights
        if evaluationData.isna().all().all() or (evaluationData == 0).all().all():
            print("No data available for evaluation.")
            return pd.Series(0, index=evaluationData.index)
        portfolioWeights = self.optimal_weights
        subperiodReturns = []
        subperiodWeights = [portfolioWeights]
        for singleSubperiodReturns in evaluationData.values:
            portfolioReturns = subperiodWeights[-1] @ singleSubperiodReturns
            portfolioWeights = subperiodWeights[-1] * (1 + singleSubperiodReturns) / (1 + portfolioReturns)
            subperiodReturns.append(portfolioReturns)
            subperiodWeights.append(portfolioWeights)
        self.actual_returns = pd.Series(subperiodReturns, index=evaluationData.index)
        self.actual_weights = pd.DataFrame(subperiodWeights[:-1], index=evaluationData.index, columns=self.ticker)
        return pd.Series(subperiodReturns, index=evaluationData.index)

spinner = Spinner("Starting...")
spinner.start()
spinner.message("Loading data...", "blue")


root = os.path.dirname(__file__)
staticPath = os.path.join(root, 'data', 'Static.xlsx')
ritPath = os.path.join(root, 'data', 'DS_RI_T_USD_M.xlsx')
rfPath = os.path.join(root, 'data', 'Risk_Free_Rate.xlsx')

staticData = pd.read_excel(staticPath, engine='openpyxl')
masterData = pd.read_excel(ritPath, usecols=lambda x: x != 'NAME', index_col=0, engine='openpyxl').transpose()
masterData.index.rename('DATE', inplace=True) # print(sum(masterData.isna().any())) # Prices have no missing values
masterData = masterData[masterData.index.year > 2000]

global masterIndex
masterIndex = masterData.index
df_dict = {}
for region in ['AMER', 'EM', 'EUR', 'PAC']:
    filter = staticData['ISIN'][staticData['Region'] == region]
    df_dict[region] = masterData[filter].pct_change()
equity_returns = pd.concat(df_dict.values(), keys=df_dict.keys(), axis=1)

commodities = {
    "Gold": "GC=F",
    "Silver": "SI=F",}

metal_returns = pd.DataFrame(index=masterIndex)
for name, ticker in commodities.items():
    data = yf.download(ticker, start=min(masterIndex), end=max(masterIndex), interval='1d')
    data_filled = data['Close'].reindex(masterIndex).ffill()
    metal_returns[name] = data_filled.pct_change()

for df in [equity_returns, metal_returns]:
    df.replace([np.inf, -np.inf, -1], np.nan, inplace=True)
    df.fillna(0, inplace=True)

portfolio_keys = ['equity_amer', 'equity_em', 'equity_pac', 'equity_eur', 'metals']
portfolio_returns = pd.DataFrame(index=masterIndex, columns=[*portfolio_keys, 'ERC'])
portfolio_returns[:] = 0

def iteration_depth(limit=None):
    if limit is None:
        YYYY = 2021
    else:
        YYYY = limit
    indexIterator = {0: {'optimizationIndex': masterIndex.year < 2006, 'evaluationIndex': masterIndex.year == 2006}}
    for year, index in zip(range(2007, YYYY + 1), range(1, 22 + 1)):
        optimizationIndex = (masterIndex.year < year) & (masterIndex.year >= 2000 + index)
        evaluationIndex = masterIndex.year == year
        indexIterator[index] = {'optimizationIndex': optimizationIndex, 'evaluationIndex': evaluationIndex}
    return indexIterator

indexIterator = iteration_depth()
spinner.message('Optimizing', 'yellow')

start_time = time.time()
for step in indexIterator:
    
    spinner.erase()
    spinner.message(f'Optimizing {step+1}/{len(indexIterator)}...', 'yellow')

    optimizationIndex = indexIterator[step]['optimizationIndex']
    evaluationIndex = indexIterator[step]['evaluationIndex']

    sampleEquity = equity_returns.loc[optimizationIndex]
    sampleMetals = metal_returns.loc[optimizationIndex]
    evaluationEquity = equity_returns.loc[evaluationIndex]
    evaluationMetals = metal_returns.loc[evaluationIndex]

    nullFilter = create_filter_mask(sampleEquity)
    sampleEquity = sampleEquity.drop(columns=nullFilter)
    evaluationEquity = evaluationEquity.drop(columns=nullFilter)

    # Equity and Commodities Portfolios
    equityPortfolioAMER = Portfolio(sampleEquity['AMER'], 'max_sharpe')
    equityPortfolioEM = Portfolio(sampleEquity['EM'], 'max_sharpe')
    equityPortfolioEUR = Portfolio(sampleEquity['EUR'], 'max_sharpe')
    equityPortfolioPAC = Portfolio(sampleEquity['PAC'], 'max_sharpe')
    metalsPortfolio = Portfolio(sampleMetals, 'max_sharpe')

    portfolio_returns.loc[evaluationIndex, 'equity_amer'] = equityPortfolioAMER.evaluate_performance(evaluationEquity['AMER']).values
    portfolio_returns.loc[evaluationIndex, 'equity_em'] = equityPortfolioEM.evaluate_performance(evaluationEquity['EM']).values
    portfolio_returns.loc[evaluationIndex, 'equity_eur'] = equityPortfolioEUR.evaluate_performance(evaluationEquity['EUR']).values
    portfolio_returns.loc[evaluationIndex, 'equity_pac'] = equityPortfolioPAC.evaluate_performance(evaluationEquity['PAC']).values
    portfolio_returns.loc[evaluationIndex, 'metals'] = metalsPortfolio.evaluate_performance(evaluationMetals).values

    # ERC Portfolio
    samplePortfolio = portfolio_returns.loc[optimizationIndex]
    evaluationPortfolio = portfolio_returns.loc[evaluationIndex]

    ercPortfolio = Portfolio(samplePortfolio[portfolio_keys], 'erc', trust_markowitz=False)

    portfolio_returns.loc[evaluationIndex, 'ERC'] = ercPortfolio.evaluate_performance(evaluationPortfolio[portfolio_keys]).values

    # print(ercPortfolio.actual_weights.head(1))
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
# default: Optimization took 12.272448062896729
# warm-start: Optimization took 12.01345682144165


# Trust Markowitz
# [ALL]: min_var - erc; 4.653315
# {'mu': 0.07657150617814579, 'std': 0.08819977139383238, 'SR': 0.8681599166083583, 'min': -0.1353098477599214, 'max': 0.07905169449900758}
# [ALL]: max_sharpe - erc; 6.090455
# {'mu': 0.09188534292840678, 'std': 0.10682250633445015, 'SR': 0.8601683866199819, 'min': -0.18966399887745636, 'max': 0.11284597357735507}

# Trust Sample
# [ALL]: min_var - erc; 4.780630
# {'mu': 0.07743546978041094, 'std': 0.08308139448015016, 'SR': 0.93204345286852**, 'min': -0.14875754901062582, 'max': 0.07565324681292446}
# [ALL]: max_sharpe - erc; 5.73339 
# {'mu': 0.08803843383025822, 'std': 0.09935060247694187, 'SR': 0.8861389023855283, 'min': -0.18885568544321116, 'max': 0.1014789952872169}

# sr-erc
# Wrapped --- 5.94755  7.670594   4.017071   5.452663  3.689266  5.73339
# Unwrapped - 5.94755  7.670594   4.017071   5.452663  3.689266  5.73339

# mv-erc 
# Sampled --- 4.669425  8.984862   3.859747   2.291611  3.694905  4.780630 - 'SR': 0.931980528997611
# EUR res --- 4.669425  8.984862   3.859747   2.744678  3.694905  4.969688 - 'SR': 0.937653663434709
# All res --- 6.120157  8.439431   3.722466   2.744678  3.694905  5.059350 - 'SR': 0.8817597034161403

# Sample moments for underlying portfolios

# [AMER]: max_sharpe - max_sharpe 
# {'mu': 0.09172913947106487, 'std': 0.11560761199805415, 'SR': 0.7934524196608161, 'min': -0.17830480979768404, 'max': 0.10239193131569116}
# {'mu': 0.07273869221370122, 'std': 0.15069922337564170, 'SR': 0.4826746321869788, 'min': -0.18005035714502649, 'max': 0.23270087979011667}
# {'mu': 0.07463570019195132, 'std': 0.12097513819898459, 'SR': 0.6169507330439055, 'min': -0.17877619218566382, 'max': 0.20070600163444913}

# [AMER]: max_sharpe - erc 
# {'mu': 0.09172913947106487, 'std': 0.11560761199805415, 'SR': 0.7934524196608161, 'min': -0.17830480979768404, 'max': 0.10239193131569116}
# {'mu': 0.07273869221370122, 'std': 0.15069922337564170, 'SR': 0.4826746321869788, 'min': -0.18005035714502649, 'max': 0.23270087979011667}
# {'mu': 0.08342414051025582, 'std': 0.10081397597045466, 'SR': 0.8275057074894532, 'min': -0.17895795287808525, 'max': 0.09431387517176634}
# [ALL]: -//-; 5.698838**
# {'mu': 0.08767184687355978, 'std': 0.09873028216153070, 'SR': 0.8879934803601753, 'min': -0.18886112098275987, 'max': 0.10135315180947321}

# [AMER]: min_var - erc
# {'mu': 0.07834498581821125, 'std': 0.10367413725804024, 'SR': 0.7556849556723498, 'min': -0.14917328462135587, 'max': 0.08445504842876275}
# {'mu': 0.07245860735849607, 'std': 0.14869978988288443, 'SR': 0.4872811684237368, 'min': -0.18005035714502649, 'max': 0.23270087979011667}
# {'mu': 0.07432796026078980, 'std': 0.09298803902345576, 'SR': 0.7993281828649054, 'min': -0.15987648111698566, 'max': 0.10890513608647240}
# [ALL]: -//-; 4.780630
# {'mu': 0.07743546978041094, 'std': 0.08308139448015016, 'SR': 0.93204345286852**, 'min': -0.14875754901062582, 'max': 0.07565324681292446}

# [AMER]: min_var - min_var
# {'mu': 0.07834498581821125, 'std': 0.10367413725804024, 'SR': 0.7556849556723498, 'min': -0.14917328462135587, 'max': 0.08445504842876275}
# {'mu': 0.07245860735849607, 'std': 0.14869978988288443, 'SR': 0.4872811684237368, 'min': -0.18005035714502649, 'max': 0.23270087979011667}
# {'mu': 0.07473273168119099, 'std': 0.09294352312780758, 'SR': 0.8040660517938969, 'min': -0.15454205642659236, 'max': 0.08815727630548274}

# [AMER]: erc - erc 
# {'mu': 0.09259233393955868, 'std': 0.12803295336865775, 'SR': 0.7231914245776129, 'min': -0.17928858736905143, 'max': 0.12870016257932135}
# {'mu': 0.07553758995855087, 'std': 0.17761332870821120, 'SR': 0.4252923500051419, 'min': -0.18742611187346070, 'max': 0.21070192097875076}
# {'mu': 0.08375917024128232, 'std': 0.11789237374745626, 'SR': 0.7104714883484105, 'min': -0.18167581812904670, 'max': 0.11614137125549948}

# [AMER]: erc - max_sharpe 
# {'mu': 0.09259233393955868, 'std': 0.12803295336865775, 'SR': 0.7231914245776129, 'min': -0.17928858736905143, 'max': 0.12870016257932135}
# {'mu': 0.07553758995855087, 'std': 0.17761332870821120, 'SR': 0.4252923500051419, 'min': -0.18742611187346070, 'max': 0.21070192097875076}
# {'mu': 0.07088341017652122, 'std': 0.13161653380544980, 'SR': 0.5385600739288439, 'min': -0.18130181186433206, 'max': 0.15288206330228960}