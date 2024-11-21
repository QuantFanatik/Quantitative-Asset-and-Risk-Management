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
import glob
import concurrent.futures


global ANNUALIZATION_FACTOR
ANNUALIZATION_FACTOR = 12

root = os.path.dirname(__file__)
staticPath = os.path.join(root, 'data', 'Static.xlsx')
ritPath = os.path.join(root, 'data', 'DS_RI_T_USD_M.xlsx')
mvPath = os.path.join(root, 'data', 'DS_MV_USD_M.xlsx')
rfPath = os.path.join(root, 'data', 'Risk_Free_Rate.xlsx')


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


from concurrent.futures import ThreadPoolExecutor

def optimize_and_evaluate_all(step, optimizationIndex, evaluationIndex):
    """
    Function to optimize and evaluate all portfolios for a given step.
    """
    sampleEquity = equity_returns.loc[optimizationIndex]
    sampleMarketValues = market_values.loc[optimizationIndex]
    sampleMetals = {key: df.loc[optimizationIndex] for key, df in aligned_returns.items()}
    evaluationEquity = equity_returns.loc[evaluationIndex]
    evaluationMetals = {key: df.loc[evaluationIndex] for key, df in aligned_returns.items()}

    # Apply filters
    minMarketCapThreshold = 0
    maxMarketCapThreshold = np.inf
    nullFilter = create_filter_mask(sampleEquity, sampleMarketValues, minMarketCapThreshold, maxMarketCapThreshold)
    sampleEquity = sampleEquity.drop(columns=nullFilter)
    evaluationEquity = evaluationEquity.drop(columns=nullFilter)

    # Portfolio definitions
    portfolio_definitions = [
        ('equity_amer', sampleEquity['AMER'], evaluationEquity['AMER']),
        ('equity_em', sampleEquity['EM'], evaluationEquity['EM']),
        ('equity_eur', sampleEquity['EUR'], evaluationEquity['EUR']),
        ('equity_pac', sampleEquity['PAC'], evaluationEquity['PAC']),
        ('Metals', sampleMetals['Metals'] if 'Metals' in sampleMetals else None, evaluationMetals['Metals'] if 'Metals' in evaluationMetals else None),
        ('Commodities', sampleMetals['Commodities'] if 'Commodities' in sampleMetals else None, evaluationMetals['Commodities'] if 'Commodities' in evaluationMetals else None),
        ('Crypto', sampleMetals['Crypto'] if 'Crypto' in sampleMetals else None, evaluationMetals['Crypto'] if 'Crypto' in evaluationMetals else None),
        ('Volatilities', sampleMetals['Volatilities'] if 'Volatilities' in sampleMetals else None, evaluationMetals['Volatilities'] if 'Volatilities' in evaluationMetals else None)
    ]

    results = {}
    for portfolio_name, sample_data, evaluation_data in portfolio_definitions:
        if sample_data is not None and not sample_data.empty:
            portfolio = Portfolio(sample_data, 'max_sharpe')
            optimization_performance = portfolio.evaluate_performance(sample_data)
            if evaluation_data is not None and not evaluation_data.empty:
                evaluation_performance = portfolio.evaluate_performance(evaluation_data)
            else:
                print(f"No evaluation data for {portfolio_name} at step {step}")
                evaluation_performance = pd.Series(0, index=evaluation_data.index if evaluation_data is not None else [])
            results[portfolio_name] = {
                'optimization': optimization_performance,
                'evaluation': evaluation_performance,
                'portfolio': portfolio
            }
        else:
            print(f"No sample data for {portfolio_name} at step {step}")
    return results

def generate_aligned_returns(path, folder_names, master_index):
    """
    Generate a DataFrame containing aligned returns from CSV files in specified folders,
    with prices forward-filled to the end of each month and aligned with master_index.

    Parameters:
    path (str): The root directory path.
    folder_names (list): A list of folder names containing the data CSV files.
    master_index (pd.Index): The index to align the data.

    Returns:
    dict: A dictionary where keys are folder names (subportfolios) and values are DataFrames of aligned returns.
    """
    subportfolios = {}

    for folder_name in folder_names:
        full_path = os.path.join(path, 'data', 'data_YF', folder_name)
        csv_files = glob.glob(os.path.join(full_path, "*.csv"))
        aligned_prices = pd.DataFrame(index=master_index)  # DataFrame to hold aligned prices for the folder

        for file in csv_files:
            filename = os.path.basename(file).replace('.csv', '')  # Extract the base name without '.csv'
            data = pd.read_csv(file, parse_dates=['Date'], index_col='Date')  # Assuming 'Date' is the date column

            # Convert index to string and trim the datetime part, keeping only YYYY-MM-DD
            data.index = data.index.astype(str).str.slice(0, 10)
            data.index = pd.to_datetime(data.index)  # Convert back to datetime format

            # Forward-fill prices for each month until the end of the month
            data = data.resample('M').ffill()

            # Align the data with master_index by reindexing and forward-filling missing values
            aligned_data = data.reindex(master_index).ffill()

            if 'Close' in aligned_data.columns:
                # Store aligned prices in the aligned_prices DataFrame
                aligned_prices[filename] = aligned_data['Close']
            else:
                print(f"Warning: 'Close' column not found in {file}. Skipping...")

        # Compute percentage changes (returns) after aligning the prices
        returns = aligned_prices.pct_change()
        returns.replace([np.inf, -np.inf], np.nan, inplace=True)
        returns.fillna(0, inplace=True)

        # Store the returns in the subportfolios dictionary
        subportfolios[folder_name] = returns

    return subportfolios

def add_tickers(path, folder_names, global_tickers):
    """
    Dynamically add tickers based on the files in the specified directories.

    Parameters:
    path (str): The root directory path.
    folder_names (list): A list of folder names containing data CSV files.
    global_tickers (list): The list to which the new tickers should be added.

    Returns:
    list: Updated list of global tickers.
    """
    for folder_name in folder_names:
        full_path = os.path.join(path, 'data', 'data_YF', folder_name)
        csv_files = glob.glob(os.path.join(full_path, "*.csv"))

        for file in csv_files:
            filename = os.path.basename(file).replace('.csv', '')  # Extract the base name without '.csv'
            if filename not in global_tickers:
                global_tickers.append(filename)

    return global_tickers



def import_additional_data(path, folder_name):
    """
    This function reads all CSV files in a specified folder under the given path.

    Parameters:
    path (str): The root directory path.
    folder_name (str): The name of the subfolder containing the CSV files.

    Returns:
    dict: A dictionary containing DataFrames for each CSV file.
    """
    full_path = os.path.join(path, 'data', 'data_YF', folder_name)
    csv_files = glob.glob(os.path.join(full_path, "*.csv"))  # Find all CSV files in the directory
    data_dict = {}  # Dictionary to store DataFrames

    for file in csv_files:
        filename = os.path.basename(file).replace('.csv', '')
        data_dict[filename] = pd.read_csv(file)

    return data_dict


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

def create_filter_mask(sampleData, marketValuesData, minMarketCap: float = -np.inf, maxMarketCap: float = np.inf):
    # Identify the latest date in both sampleData and marketValuesData
    latestDateSample = sampleData.index.max()

    # Zero December Returns filter (activated/deactivated based on criteria)
    decemberData = sampleData.loc[[latestDateSample]]
    decemberFilter = decemberData.columns[decemberData.iloc[0] == np.inf]  # deactivated

    # December price below threshold filter
    yearEndPrices = sampleData.loc[latestDateSample]
    priceFilter = yearEndPrices[yearEndPrices < -np.inf].index  # activated

    # High return filter (both high and low extremes)
    returnFilterHigh = sampleData.columns[sampleData.max() >= np.inf]  # deactivated
    returnFilterLow = sampleData.columns[sampleData.min() <= -np.inf]  # deactivated
    returnFilter = returnFilterHigh.union(returnFilterLow)
    
    # Frequent Zero Returns filter
    startOfYear = pd.Timestamp(latestDateSample.year, 1, 1)
    yearlyData = sampleData.loc[startOfYear:latestDateSample]
    monthsWithZeroReturns = (yearlyData == 0).sum(axis=0)
    frequentZerosFilter = monthsWithZeroReturns[monthsWithZeroReturns >= 12].index  # activated

    # Market Cap filters based on the latest date in marketValuesData
    marketValuesAtEnd = marketValuesData.loc[latestDateSample]
    marketCapFilterMin = marketValuesAtEnd[marketValuesAtEnd < minMarketCap].index
    marketCapFilterMax = marketValuesAtEnd[marketValuesAtEnd > maxMarketCap].index

    # Combine all filters
    combinedFilter = decemberFilter.union(frequentZerosFilter).union(priceFilter).union(returnFilter)
    combinedFilter = combinedFilter.union(marketCapFilterMin).union(marketCapFilterMax)
    
    # Return the combined filter
    return combinedFilter

class Portfolio():
    valid_types = ('markowitz', 'erc', 'max_sharpe', 'min_var')
    non_combined_portfolios = []
    gamma_linspace = np.linspace(-0.5, 1.5, 101)
    
    def __init__(self, returns: pd.DataFrame | pd.Series, type: str='markowitz', names: list[str]=None, trust_markowitz: bool=False, resample: bool=False, target_gamma=None):
        assert type.lower() in self.valid_types, f"Invalid type: {type}. Valid types are: {self.valid_types}"
        #TODO: Attention! ERC portfolios use sample returns, not ex-ante expectations.
        self.trust_markowitz = trust_markowitz
        self.gamma = self.assign_gamma(target_gamma)
        self.resample = resample
        self.type = type.lower()
        self.ticker = returns.columns
        self.returns = returns
        self.expected_returns = self.get_expected_returns()
        self.expected_covariance = self.get_expected_covariance()
        self.dim = len(self.expected_returns)
        self.len = len(self.returns)

        self.frontier = ... # Frontier is calulated in get_optimal()
        self.optimal_weights = self.get_optimal()
        self.expected_portfolio_return = self.get_expected_portfolio_return()
        self.expected_portfolio_varcov = self.get_expected_portfolio_varcov()

        if self.type != 'erc':
            Portfolio.non_combined_portfolios.append(self)
        if self.type == 'erc':
            self.frontier.loc[0, 'expected_return'] = self.expected_portfolio_return
            self.frontier.loc[0, 'expected_variance'] = self.expected_portfolio_varcov
            self.frontier.loc[0, 'expected_sharpe'] = self.expected_portfolio_return / np.sqrt(self.expected_portfolio_varcov)
            for i, asset in enumerate(self.ticker):
                self.frontier.loc[0, asset] = self.optimal_weights[i]

    def assign_gamma(self, gamma):
        """Assigns the closest gamma value from gamma_linspace to the provided target_gamma."""
        if gamma is None:
            return None
        return min(self.gamma_linspace, key=lambda x: abs(x - gamma))

    def get_optimal(self):
        self.frontier = self.get_frontier()
        if self.type == 'markowitz':
            assert self.gamma is not None, "Markowitz optimization requires a gamma value."
            return self.frontier.loc[self.gamma, self.ticker].values
        if self.type == 'min_var':
            return self.frontier.loc[self.frontier['expected_variance'].idxmin(), self.ticker].values
        if self.type == 'max_sharpe':
            return self.frontier.loc[self.frontier['expected_sharpe'].idxmax(), self.ticker].values
        if self.type == 'erc':
            return self._fit_erc()

    def get_frontier(self, singular=None):
        """Calculate the efficient frontier."""
        method = self._frontier_method()
        if self.resample:
            frontier_weights = self._resample(method)
        else:
            frontier_weights = method(singular)
        return self._pickle_frontier(frontier_weights, singular)
    
    def _frontier_method(self):
        if self.dim >= 30:
            return self._efficient_frontier_cvxpy
        else:
            return self._efficient_frontier_scipy
        
    def _pickle_frontier(self, frontier_weights: np.ndarray, singular=None) -> pd.DataFrame:
        """Helper method to create a DataFrame from the efficient frontier weights."""
        if singular is not None:
            frontier_weights = frontier_weights.reshape(1, -1)

        expected_returns_vector = frontier_weights @ self.expected_returns
        expected_variances_vector = np.einsum('ij,jk,ik->i', frontier_weights, self.expected_covariance, frontier_weights)
        
        expected_sharpe = np.zeros_like(expected_returns_vector)
        non_zero_variance = expected_variances_vector > 0
        expected_sharpe[non_zero_variance] = (expected_returns_vector[non_zero_variance] / np.sqrt(expected_variances_vector[non_zero_variance]))

        data = {
            'gamma': self.gamma_linspace if not singular else [singular],
            'expected_return': expected_returns_vector,
            'expected_variance': expected_variances_vector,
            'expected_sharpe': expected_sharpe}
        
        weight_columns = {f'{asset}': frontier_weights[:, i] for i, asset in enumerate(self.ticker)}
        data.update(weight_columns)

        frontier_df = pd.DataFrame(data)
        frontier_df.set_index('gamma', inplace=True)        
        return frontier_df
        
    def _efficient_frontier_scipy(self, singular=None):
        initial_guess = np.ones(self.dim) / self.dim 
        constraints = [LinearConstraint(np.ones(self.dim), 1, 1)]
        bounds = Bounds(0, 1)

        if not singular:
            results = np.zeros((len(self.gamma_linspace), self.dim))
            itterator = enumerate(self.gamma_linspace)
        else:
            itterator = [(0, singular)]

        for i, gamma in itterator:
            def objective(weights):
                return 0.5 * np.dot(weights.T, np.dot(self.expected_covariance, weights)) - gamma * np.dot(self.expected_returns, weights)
            def jacobian(weights):
                return np.dot(self.expected_covariance, weights) - gamma * self.expected_returns

            kwargs = {'fun': objective,
                      'jac': jacobian,
                      'x0': initial_guess,
                      'constraints': constraints,
                      'bounds': bounds,
                      'method': 'SLSQP',
                      'tol': 1e-16}
            result = minimize(**kwargs)

            if not singular:
                results[i, :] = result.x
                initial_guess = result.x
            else:
                results=result.x
        return results

    def _efficient_frontier_cvxpy(self, singular=None):
        weights = cp.Variable(self.dim)
        gamma_param = cp.Parameter(nonneg=False)
        markowitz = 0.5 * cp.quad_form(weights, cp.psd_wrap(self.expected_covariance)) - gamma_param * self.expected_returns.T @ weights
        constraints = [cp.sum(weights) == 1, weights >= 0]
        problem = cp.Problem(cp.Minimize(markowitz), constraints)

        if not singular:
            results = np.zeros((len(self.gamma_linspace), self.dim))
            itterator = enumerate(self.gamma_linspace)
        else:
            itterator = [(0, singular)]

        for i, gamma in itterator:
            gamma_param.value = gamma
            problem.solve(warm_start=True)
            if not singular:
                results[i, :] = weights.value
            else:
                results = weights.value
        return results

    def get_expected_returns(self) -> pd.DataFrame | pd.Series:
        #TODO: Attention! If extending beyond ERC, if statement must be updated.
        if self.trust_markowitz and self.type == 'erc':
            internal_expectations = np.array([portfolio.expected_portfolio_return for portfolio in Portfolio.non_combined_portfolios])
            return pd.Series(internal_expectations, index=self.returns.columns)
        return self.returns.mean(axis=0)
    
    def get_expected_covariance(self) -> pd.DataFrame | pd.Series:
        if self.trust_markowitz and self.type == 'erc':
            internal_expectations = np.array([np.sqrt(portfolio.expected_portfolio_varcov) for portfolio in Portfolio.non_combined_portfolios])
            sample_correlations = self.returns.corr().fillna(0)
            varcov_matrix = np.outer(internal_expectations, internal_expectations) * sample_correlations
            return pd.DataFrame(varcov_matrix, index=self.returns.columns, columns=self.returns.columns)
        return self.returns.cov(ddof=0)
    
    def get_expected_portfolio_return(self) -> float:
        return np.dot(self.expected_returns, self.optimal_weights)
    
    def get_expected_portfolio_varcov(self) -> float:
        return self.optimal_weights.T @ self.expected_covariance @ self.optimal_weights

    def _resample(self, method) -> np.ndarray:
        #TODO: Attention! Low number of simulations set for testing
        N_SUMULATIONS = 2 # 500

        original_moments = (self.expected_returns.copy(), self.expected_covariance.copy())
        simulated_weights = []

        for i in range(N_SUMULATIONS):
            np.random.seed(i)
            simulated_returns = np.random.multivariate_normal(self.expected_returns, self.expected_covariance, self.len)
            # TODO: verify necessity of annualization factor
            self.expected_returns = self._pandify(np.mean(simulated_returns, axis=0))# * ANNUALIZATION_FACTOR
            self.expected_covariance = self._pandify(np.cov(simulated_returns.T, ddof=0))
            self.optimal_weights = method()
            simulated_weights.append(self.optimal_weights)
        
        self.expected_returns, self.expected_covariance = original_moments
        combined_simulation_data = np.stack(simulated_weights, axis=0)
        return combined_simulation_data.mean(axis=0) # mean across gammas
    
    def _pandify(self, array: np.ndarray) -> pd.Series | pd.DataFrame:
        if array.ndim == 1:
            return pd.Series(array, index=self.ticker)
        else:
            return pd.DataFrame(array, index=self.ticker, columns=self.ticker)
    
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
    
    def log_visuals(self):
        gammas = np.linspace(-0.5, 1.5, 101)
        efficient_frontier = self.__class__.efficient_frontier(gammas, self.expected_returns, self.expected_covariance)
        return efficient_frontier
        
spinner = Spinner("Starting...")
spinner.start()
spinner.message("Loading data...", "blue")


root = os.path.dirname(__file__)
staticPath = os.path.join(root, 'data', 'Static.xlsx')
ritPath = os.path.join(root, 'data', 'DS_RI_T_USD_M.xlsx')
mvPath = os.path.join(root, 'data', 'DS_MV_USD_M.xlsx')
rfPath = os.path.join(root, 'data', 'Risk_Free_Rate.xlsx')

staticData = pd.read_excel(staticPath, engine='openpyxl')
masterData = pd.read_excel(ritPath, usecols=lambda x: x != 'NAME', index_col=0, engine='openpyxl').transpose()
masterData.index.rename('DATE', inplace=True) # print(sum(masterData.isna().any())) # Prices have no missing values
masterData = masterData[masterData.index.year > 2000]

capData = pd.read_excel(mvPath, usecols=lambda x: x != 'NAME', index_col=0, engine='openpyxl').transpose()
capData.index = pd.to_datetime(capData.index, format='%Y-%m-%d')
capData.index.rename('DATE', inplace=True)
capData = capData[capData.index.year > 2000] * 1e6

global masterIndex, global_tickers
masterIndex = masterData.index
global_tickers = list(masterData.columns)
df_dict = {}
mv_dict = {}
for region in ['AMER', 'EM', 'EUR', 'PAC']:
    filter = staticData['ISIN'][staticData['Region'] == region] # filter is the isin for a given region
    df_dict[region] = masterData[filter].pct_change() # now we have a dictionnary with some retunrs dataset for a given region
    mv_dict[region] = capData[filter]

print(df_dict)
equity_returns = pd.concat(df_dict.values(), keys=df_dict.keys(), axis=1)
print(equity_returns)
market_values = pd.concat(mv_dict.values(), keys=mv_dict.keys(), axis=1)

"""
commodities = {
    "Gold": "GC=F",
    "Silver": "SI=F",}

#print(global_tickers)

global_tickers.extend(commodities.keys()) # all the tickers + gold and silver
"""


# Example usage
folders = ['Metals', 'Commodities', 'Volatilities', 'Crypto']  # List of folders
global_tickers = add_tickers(root, folders, global_tickers)  # Add tickers from all folders

# Generate aligned returns for all subportfolios
aligned_returns = generate_aligned_returns(root, folders, masterIndex)


print(aligned_returns)

# Print the returns for each subportfolio
for folder, returns in aligned_returns.items():
    print(f"Aligned returns for {folder}:")
    print(returns.head())

print(global_tickers)
"""
metal_returns = generate_aligned_metal_returns(root, metals_folder, masterIndex)

# Now metal_returns is ready to be used for further processing and portfolio optimization
print(metal_returns.head())
"""

for key, df in aligned_returns.items():
    df.replace([np.inf, -np.inf, -1], np.nan, inplace=True)
    df.fillna(0, inplace=True)

# Ensure `equity_returns` is treated correctly as a DataFrame
equity_returns.replace([np.inf, -np.inf, -1], np.nan, inplace=True)
equity_returns.fillna(0, inplace=True)

portfolio_keys = ['equity_amer', 'equity_em', 'equity_pac', 'equity_eur', 'Metals', 'Commodities', 'Volatilities', 'Crypto']
portfolio_returns = pd.DataFrame(index=masterIndex, columns=[*portfolio_keys, 'ERC'])
portfolio_returns[:] = 0

global_tickers.extend(portfolio_keys) # all the tickers + gold and silver + 'equity_amer', 'equity_em', 'equity_pac', 'equity_eur', 'metals'
#print(global_tickers)
def iteration_depth(limit=None, frequency="annual"):
    if frequency == "annual":
        if limit is None:
            YYYY = 2021
        else:
            YYYY = limit
        indexIterator = {0: {'optimizationIndex': masterIndex.year < 2006, 'evaluationIndex': masterIndex.year == 2006}}
        for year, index in zip(range(2007, YYYY + 1), range(1, 22 + 1)):
            optimizationIndex = (masterIndex.year < year) & (masterIndex.year >= 2000 + index)
            evaluationIndex = masterIndex.year == year
            indexIterator[index] = {'optimizationIndex': optimizationIndex, 'evaluationIndex': evaluationIndex}

    elif frequency == "monthly":
        if limit is None:
            YYYY = 2021
        else:
            YYYY = limit
        index = 0
        indexIterator = {}
        for year in range(2006, YYYY + 1):
            for month in range(1, 13):
                if year == YYYY and month > 1:
                    break
                # Calculate 5-year (60 months) rolling lookback period
                start_year = year
                start_month = month - 1
                if start_month == 0:
                    start_month = 12
                    start_year -= 1
                end_year = start_year - 5
                end_month = start_month

                optimizationIndex = (
                    ((masterIndex.year > end_year) |
                    ((masterIndex.year == end_year) & (masterIndex.month >= end_month))) &
                    ((masterIndex.year < year) |
                    ((masterIndex.year == year) & (masterIndex.month < month)))
                )
                evaluationIndex = (masterIndex.year == year) & (masterIndex.month == month)
                indexIterator[index] = {'optimizationIndex': optimizationIndex, 'evaluationIndex': evaluationIndex}
                index += 1
    return indexIterator


visual_data = {}
indexIterator = iteration_depth(frequency="annual")
print(indexIterator)
spinner.message('Optimizing', 'yellow')

"""
start_time = time.time()

for step in indexIterator:

    spinner.erase()
    spinner.message(f'Optimizing {step+1}/{len(indexIterator)}...', 'yellow')

    optimizationIndex = indexIterator[step]['optimizationIndex']
    evaluationIndex = indexIterator[step]['evaluationIndex']

    sampleEquity = equity_returns.loc[optimizationIndex]
    sampleMarketValues = market_values.loc[optimizationIndex]  # Filter data based on optimizationIndex

    # Handle aligned_returns as a dictionary
    sampleMetals = {key: df.loc[optimizationIndex] for key, df in aligned_returns.items()}
    evaluationEquity = equity_returns.loc[evaluationIndex]
    evaluationMetals = {key: df.loc[evaluationIndex] for key, df in aligned_returns.items()}


    # Apply filters
    minMarketCapThreshold = 0
    maxMarketCapThreshold = np.inf
    nullFilter = create_filter_mask(sampleEquity, sampleMarketValues, minMarketCapThreshold, maxMarketCapThreshold)
    sampleEquity = sampleEquity.drop(columns=nullFilter)
    evaluationEquity = evaluationEquity.drop(columns=nullFilter)
    print(sampleEquity.shape)

    # Equity Portfolios
    equityPortfolioAMER = Portfolio(sampleEquity['AMER'], 'max_sharpe')
    equityPortfolioEM = Portfolio(sampleEquity['EM'], 'max_sharpe')
    equityPortfolioEUR = Portfolio(sampleEquity['EUR'], 'max_sharpe')
    equityPortfolioPAC = Portfolio(sampleEquity['PAC'], 'max_sharpe')

    # Metals and Other Portfolios
    metalsPortfolio = Portfolio(sampleMetals['Metals'], 'max_sharpe') if 'Metals' in sampleMetals else None
    cryptoPortfolio = Portfolio(sampleMetals['Crypto'], 'max_sharpe') if 'Crypto' in sampleMetals else None
    commoditiesPortfolio = Portfolio(sampleMetals['Commodities'], 'max_sharpe') if 'Commodities' in sampleMetals else None
    volatilitiesPortfolio = Portfolio(sampleMetals['Volatilities'], 'max_sharpe') if 'Volatilities' in sampleMetals else None

    # Evaluate Portfolio Returns
    portfolio_returns.loc[optimizationIndex, 'equity_amer'] = equityPortfolioAMER.evaluate_performance(
        sampleEquity['AMER']).values
    portfolio_returns.loc[evaluationIndex, 'equity_amer'] = equityPortfolioAMER.evaluate_performance(
        evaluationEquity['AMER']).values

    portfolio_returns.loc[optimizationIndex, 'equity_em'] = equityPortfolioEM.evaluate_performance(
        sampleEquity['EM']).values
    portfolio_returns.loc[evaluationIndex, 'equity_em'] = equityPortfolioEM.evaluate_performance(
        evaluationEquity['EM']).values

    portfolio_returns.loc[optimizationIndex, 'equity_eur'] = equityPortfolioEUR.evaluate_performance(
        sampleEquity['EUR']).values
    portfolio_returns.loc[evaluationIndex, 'equity_eur'] = equityPortfolioEUR.evaluate_performance(
        evaluationEquity['EUR']).values

    portfolio_returns.loc[optimizationIndex, 'equity_pac'] = equityPortfolioPAC.evaluate_performance(
        sampleEquity['PAC']).values
    portfolio_returns.loc[evaluationIndex, 'equity_pac'] = equityPortfolioPAC.evaluate_performance(
        evaluationEquity['PAC']).values

    if metalsPortfolio:
        portfolio_returns.loc[optimizationIndex, 'Metals'] = metalsPortfolio.evaluate_performance(
            sampleMetals['Metals']).values
        portfolio_returns.loc[evaluationIndex, 'Metals'] = metalsPortfolio.evaluate_performance(
            evaluationMetals['Metals']).values

    if commoditiesPortfolio:
        portfolio_returns.loc[optimizationIndex, 'Commodities'] = commoditiesPortfolio.evaluate_performance(
            sampleMetals['Commodities']).values
        portfolio_returns.loc[evaluationIndex, 'Commodities'] = commoditiesPortfolio.evaluate_performance(
            evaluationMetals['Commodities']).values

    if cryptoPortfolio:
        portfolio_returns.loc[optimizationIndex, 'Crypto'] = cryptoPortfolio.evaluate_performance(
            sampleMetals['Crypto']).values
        portfolio_returns.loc[evaluationIndex, 'Crypto'] = cryptoPortfolio.evaluate_performance(
            evaluationMetals['Crypto']).values

    if volatilitiesPortfolio:
        portfolio_returns.loc[optimizationIndex, 'Volatilities'] = volatilitiesPortfolio.evaluate_performance(
            sampleMetals['Volatilities']).values
        portfolio_returns.loc[evaluationIndex, 'Volatilities'] = volatilitiesPortfolio.evaluate_performance(
            evaluationMetals['Volatilities']).values

    # Construct the ERC Portfolio
    samplePortfolio = portfolio_returns.loc[optimizationIndex]
    evaluationPortfolio = portfolio_returns.loc[evaluationIndex]
    ercPortfolio = Portfolio(samplePortfolio[portfolio_keys], 'erc', trust_markowitz=False)
    portfolio_returns.loc[evaluationIndex, 'ERC'] = ercPortfolio.evaluate_performance(
        evaluationPortfolio[portfolio_keys]).values
    
    print(portfolio_returns.loc[evaluationIndex])
    # Optional Visual Logging
    portfolios = [equityPortfolioAMER, equityPortfolioEM, equityPortfolioEUR, equityPortfolioPAC, metalsPortfolio, ercPortfolio]
    portfolio_names = portfolio_keys + ['ERC']
    for portfolio, portfolio_name in zip(portfolios, portfolio_names):
        if portfolio is None:
            continue  # Skip if the portfolio was not created
        tickers = portfolio.ticker
        frontier = portfolio.frontier

        expected_returns = frontier['expected_return'].values
        expected_variances = frontier['expected_variance'].values
        expected_sharpes = frontier['expected_sharpe'].values
        weights = frontier.loc[:, tickers].values

        for i, gamma in enumerate(Portfolio.gamma_linspace):
            row_data = [expected_returns[i], expected_variances[i], expected_sharpes[i]]

            weight_row = [np.nan] * len(global_tickers)
            for j, asset in enumerate(tickers):
                asset_index = global_tickers.index(asset)
                weight_row[asset_index] = weights[i, j]

            row_data.extend(weight_row)
            visual_data[(step, gamma, portfolio_name)] = row_data

    Portfolio.non_combined_portfolios = []



end_first = time.time()
print("Finished initial optimisation routine in :", (end_first- start_time))
"""
# Adjusted main code for collecting and storing results

start_second = time.time()
all_results = {}
with ThreadPoolExecutor() as executor:
    futures = {
        executor.submit(optimize_and_evaluate_all, step, indexIterator[step]['optimizationIndex'], indexIterator[step]['evaluationIndex']): step
        for step in indexIterator
    }
    for future in concurrent.futures.as_completed(futures):
        step = futures[future]
        result = future.result()
        all_results[step] = result

# Ensure that the portfolio_returns DataFrame has the correct index
portfolio_returns = portfolio_returns.reindex(masterIndex)

# Populate portfolio_returns and prepare for ERC calculation
for step, result in all_results.items():
    optimizationIndex = indexIterator[step]['optimizationIndex']
    evaluationIndex = indexIterator[step]['evaluationIndex']
    for portfolio_name, data in result.items():
        if data and 'optimization' in data and 'evaluation' in data:
            # Assign the Series directly, ensuring indices match
            #print(f"Assigning optimization data for {portfolio_name} at step {step}:")
            #print(f"Data index: {data['optimization'].index}")
            #print(f"Optimization index: {portfolio_returns.loc[data['optimization'].index].index}")
            portfolio_returns.loc[data['optimization'].index, portfolio_name] = data['optimization']
            portfolio_returns.loc[data['evaluation'].index, portfolio_name] = data['evaluation']
        else:
            print(f"Warning: Missing data for {portfolio_name} at step {step}")

    # Construct the sample and evaluation portfolios for the ERC calculation
    samplePortfolio = portfolio_returns.loc[optimizationIndex, portfolio_keys]
    evaluationPortfolio = portfolio_returns.loc[evaluationIndex, portfolio_keys]
    if not samplePortfolio.empty and not evaluationPortfolio.empty:
        ercPortfolio = Portfolio(samplePortfolio, 'erc', trust_markowitz=False)
        erc_performance = ercPortfolio.evaluate_performance(evaluationPortfolio)
        portfolio_returns.loc[erc_performance.index, 'ERC'] = erc_performance
    else:
        print(f"Warning: Empty portfolio data for ERC calculation at step {step}")

print(portfolio_returns)
end_second = time.time()
print("Finished execution time: ", (end_second - start_second))


start_third = time.time()
from concurrent.futures import ThreadPoolExecutor

for step in indexIterator:

    spinner.erase()
    spinner.message(f'Optimizing {step+1}/{len(indexIterator)}...', 'yellow')

    optimizationIndex = indexIterator[step]['optimizationIndex']
    evaluationIndex = indexIterator[step]['evaluationIndex']

    sampleEquity = equity_returns.loc[optimizationIndex]
    sampleMarketValues = market_values.loc[optimizationIndex]  # Filter data based on optimizationIndex

    # Handle aligned_returns as a dictionary
    sampleMetals = {key: df.loc[optimizationIndex] for key, df in aligned_returns.items()}
    evaluationEquity = equity_returns.loc[evaluationIndex]
    evaluationMetals = {key: df.loc[evaluationIndex] for key, df in aligned_returns.items()}

    # Apply filters
    minMarketCapThreshold = 0
    maxMarketCapThreshold = np.inf
    nullFilter = create_filter_mask(sampleEquity, sampleMarketValues, minMarketCapThreshold, maxMarketCapThreshold)
    sampleEquity = sampleEquity.drop(columns=nullFilter)
    evaluationEquity = evaluationEquity.drop(columns=nullFilter)
    print(sampleEquity.shape)

    # Portfolio definitions
    portfolio_definitions = [
        ('equity_amer', sampleEquity['AMER'], evaluationEquity['AMER']),
        ('equity_em', sampleEquity['EM'], evaluationEquity['EM']),
        ('equity_eur', sampleEquity['EUR'], evaluationEquity['EUR']),
        ('equity_pac', sampleEquity['PAC'], evaluationEquity['PAC']),
        ('Metals', sampleMetals['Metals'] if 'Metals' in sampleMetals else None, evaluationMetals['Metals'] if 'Metals' in evaluationMetals else None),
        ('Commodities', sampleMetals['Commodities'] if 'Commodities' in sampleMetals else None, evaluationMetals['Commodities'] if 'Commodities' in evaluationMetals else None),
        ('Crypto', sampleMetals['Crypto'] if 'Crypto' in sampleMetals else None, evaluationMetals['Crypto'] if 'Crypto' in evaluationMetals else None),
        ('Volatilities', sampleMetals['Volatilities'] if 'Volatilities' in sampleMetals else None, evaluationMetals['Volatilities'] if 'Volatilities' in evaluationMetals else None)
    ]

    def optimize_and_evaluate(portfolio_name, sample_data, evaluation_data):
        if sample_data is not None:
            portfolio = Portfolio(sample_data, 'max_sharpe')
            portfolio_returns.loc[optimizationIndex, portfolio_name] = portfolio.evaluate_performance(sample_data).values
            portfolio_returns.loc[evaluationIndex, portfolio_name] = portfolio.evaluate_performance(evaluation_data).values
            return portfolio_name, portfolio
        return portfolio_name, None

    # Parallel processing
    # Corrected parallel processing and collection of results
    with ThreadPoolExecutor() as executor:
        futures = {name: executor.submit(optimize_and_evaluate, name, sample, eval) for name, sample, eval in
                   portfolio_definitions}
        optimized_portfolios = {name: future.result()[1] for name, future in futures.items()}

    # Construct the ERC Portfolio
    samplePortfolio = portfolio_returns.loc[optimizationIndex]
    evaluationPortfolio = portfolio_returns.loc[evaluationIndex]
    ercPortfolio = Portfolio(samplePortfolio[portfolio_keys], 'erc', trust_markowitz=False)
    portfolio_returns.loc[evaluationIndex, 'ERC'] = ercPortfolio.evaluate_performance(
        evaluationPortfolio[portfolio_keys]).values

    print(portfolio_returns.loc[evaluationIndex])

    # Optional Visual Logging
    portfolios = [optimized_portfolios[name] for name in ['equity_amer', 'equity_em', 'equity_eur', 'equity_pac', 'Metals', 'Commodities', 'Crypto', 'Volatilities']] + [ercPortfolio]
    portfolio_names = portfolio_keys + ['ERC']
    for portfolio, portfolio_name in zip(portfolios, portfolio_names):
        if portfolio is None:
            continue  # Skip if the portfolio was not created
        tickers = portfolio.ticker
        frontier = portfolio.frontier

        expected_returns = frontier['expected_return'].values
        expected_variances = frontier['expected_variance'].values
        expected_sharpes = frontier['expected_sharpe'].values
        weights = frontier.loc[:, tickers].values

        for i, gamma in enumerate(Portfolio.gamma_linspace):
            row_data = [expected_returns[i], expected_variances[i], expected_sharpes[i]]

            weight_row = [np.nan] * len(global_tickers)
            for j, asset in enumerate(tickers):
                asset_index = global_tickers.index(asset)
                weight_row[asset_index] = weights[i, j]

            row_data.extend(weight_row)
            visual_data[(step, gamma, portfolio_name)] = row_data

    Portfolio.non_combined_portfolios = []



index = pd.MultiIndex.from_tuples(visual_data.keys(), names=["year", "gamma", "portfolio"])
columns = pd.MultiIndex.from_tuples(
    [("metrics", "expected_return"), ("metrics", "expected_variance"), ("metrics", "expected_sharpe")] +
    [("weights", asset) for asset in global_tickers],
    names=["category", "attribute"]
)

visual_df = pd.DataFrame.from_dict(visual_data, orient="index", columns=columns)
visual_df.index = index  # Set the MultiIndex
years = visual_df.index.get_level_values("year").unique()
# We must break up the file into smaller chunks to use git
for year in years:
    yearly_data = visual_df.xs(year, level="year")
    yearly_data.to_hdf(f'data/efficient_frontiers_{year}.hdf', key='frontier_data', mode='w')


# print(visual_df.loc[(slice(None), slice(None), 'equity_amer'), :].dropna(how='all', axis=1))
# print(visual_df.loc[(slice(None), slice(None), 'ERC'), :].dropna(how='all', axis=1))
    

spinner.erase()
spinner.message('Done!\n', 'green')
spinner.stop()

print((1 + portfolio_returns[portfolio_returns.index.year < 2022]).cumprod().tail(1))
# sharpe ratio of portfolios
print(portfolio_evaluation(portfolio_returns, pd.Series(0, index=portfolio_returns.index))['SR'])
# print(portfolio_evaluation(portfolio_returns['metals'], pd.Series(0, index=portfolio_returns.index)))
print(portfolio_evaluation(portfolio_returns['ERC'], pd.Series(0, index=portfolio_returns.index)))
print(f"Optimization Runtime: {(time.time() - start_third):2f}s")


