import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, minimize
import cvxpy as cp
import os
import threading
import time
from itertools import cycle
import glob
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import sys

# Global Constants
ANNUALIZATION_FACTOR = 12

# Define Paths
root = os.path.dirname(__file__)
staticPath = os.path.join(root, 'data', 'Static.xlsx')
ritPath = os.path.join(root, 'data', 'DS_RI_T_USD_M.xlsx')
mvPath = os.path.join(root, 'data', 'DS_MV_USD_M.xlsx')
rfPath = os.path.join(root, 'data', 'Risk_Free_Rate.xlsx')

# Spinner Class for Console Messages
class Spinner:
    def __init__(self, message="Processing...", color="white"):
        self.spinner = cycle(['|', '/', '-', '\\'])
        self.stop_running = threading.Event()
        self.message_text = message
        self.lock = threading.Lock()
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
                    colored_symbol = self.color_code.get(self.current_color, self.color_code["white"]) + next(
                        self.spinner) + self.color_code["reset"]
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
            colored_message = self.color_code.get(color, self.color_code["white"]) + new_message + self.color_code[
                "reset"]
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

# Function to Generate Aligned Returns
def generate_aligned_returns(path, folder_names, master_index):
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
            data = data.resample('M').ffill()  # Monthly frequency

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

# Function to Add Tickers Dynamically
def add_tickers(path, folder_names, global_tickers):
    for folder_name in folder_names:
        full_path = os.path.join(path, 'data', 'data_YF', folder_name)
        csv_files = glob.glob(os.path.join(full_path, "*.csv"))

        for file in csv_files:
            filename = os.path.basename(file).replace('.csv', '')  # Extract the base name without '.csv'
            if filename not in global_tickers:
                global_tickers.append(filename)

    return global_tickers

# Function to Load Excel Data
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

# Annualization Functions
def annualized_mean(sample_mean: float) -> float:
    return (1 + sample_mean) ** ANNUALIZATION_FACTOR - 1

def annualized_volatility(sample_std: float) -> float:
    return sample_std * np.sqrt(ANNUALIZATION_FACTOR)

def sharpe_ratio(mean: float, volatility: float) -> float:
    if volatility == 0:
        return 0
    return mean / volatility

# Portfolio Evaluation Function
def portfolio_evaluation(monthlyReturns: pd.Series | np.ndarray, monthlyRFrate: pd.Series) -> dict:
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

# Revised Filter Mask Function with Scoring
def create_filter_mask_scoring(sampleData, marketValuesData,
                               minMarketCap: float = 0,
                               maxMarketCap: float = np.inf,
                               price_threshold: float = 0.05,
                               return_threshold_high: float = 0.2,
                               return_threshold_low: float = -0.2,
                               max_filter_failures: int = 2):
    latestDateSample = sampleData.index.max()
    print(f"\nLatest Date Sample: {latestDateSample}")

    # Initialize failure count for each asset
    failure_counts = pd.Series(0, index=sampleData.columns)

    # December Returns filter
    decemberData = sampleData.loc[[latestDateSample]]
    decemberFail = decemberData.iloc[0].abs() > price_threshold
    failure_counts += decemberFail.astype(int)
    print(f"December Filter Failures (|return| > {price_threshold}): {decemberFail.sum()}")

    # Price filter
    yearEndPrices = sampleData.loc[latestDateSample]
    priceFail = yearEndPrices < price_threshold
    failure_counts += priceFail.astype(int)
    print(f"Price Filter Failures (price < {price_threshold}): {priceFail.sum()}")

    # Return filters
    returnFailHigh = sampleData.max() >= return_threshold_high
    returnFailLow = sampleData.min() <= return_threshold_low
    returnFail = returnFailHigh | returnFailLow
    failure_counts += returnFail.astype(int)
    print(
        f"Return Filter Failures (high >= {return_threshold_high} or low <= {return_threshold_low}): {returnFail.sum()}")

    # Frequent Zero Returns filter
    startOfYear = pd.Timestamp(latestDateSample.year, 1, 1)
    yearlyData = sampleData.loc[startOfYear:latestDateSample]
    monthsWithZeroReturns = (yearlyData == 0).sum(axis=0)
    frequentZerosFail = monthsWithZeroReturns >= 15  # Adjusted threshold
    failure_counts += frequentZerosFail.astype(int)
    print(f"Frequent Zeros Filter Failures (>= 15 zeros): {frequentZerosFail.sum()}")

    # Market Cap filters
    marketValuesAtEnd = marketValuesData.loc[latestDateSample]
    marketCapFailMin = marketValuesAtEnd < minMarketCap
    marketCapFailMax = marketValuesAtEnd > maxMarketCap
    failure_counts += marketCapFailMin.astype(int) + marketCapFailMax.astype(int)
    print(
        f"Market Cap Filter Failures (cap < {minMarketCap} or cap > {maxMarketCap}): {marketCapFailMin.sum() + marketCapFailMax.sum()}")

    # Assets that exceed the maximum allowed filter failures
    combinedFilter = failure_counts > max_filter_failures
    print(f"Assets exceeding max filter failures ({max_filter_failures}): {combinedFilter.sum()}")

    # Summary
    initial_asset_count = len(sampleData.columns)
    final_asset_count = initial_asset_count - combinedFilter.sum()
    print(f"Number of assets after scoring filter: {final_asset_count} out of {initial_asset_count}")

    return combinedFilter[combinedFilter].index

# Revised Portfolio Class with Enhanced Error Handling
class Portfolio():
    valid_types = ('markowitz', 'erc', 'max_sharpe', 'min_var')
    non_combined_portfolios = []
    gamma_linspace = np.linspace(-0.5, 1.5, 101)

    def __init__(self, returns: pd.DataFrame | pd.Series, type: str = 'markowitz', names: list[str] = None,
                 trust_markowitz: bool = False, resample: bool = False, target_gamma=None):
        if returns.empty:
            print(f"Warning: Initialized Portfolio with empty returns for type {type}.")
            self.type = type.lower()
            self.ticker = returns.columns if isinstance(returns, pd.DataFrame) else [returns.name]
            self.returns = returns
            self.optimal_weights = np.zeros(len(self.ticker))
            self.expected_portfolio_return = 0.0
            self.expected_portfolio_varcov = 0.0
            self.frontier = pd.DataFrame()
            # Initialize actual_weights and actual_returns
            self.actual_weights = pd.DataFrame(columns=self.ticker)
            self.actual_returns = pd.Series(dtype=float)
            return

        assert type.lower() in self.valid_types, f"Invalid type: {type}. Valid types are: {self.valid_types}"
        # TODO: Attention! ERC portfolios use sample returns, not ex-ante expectations.
        self.trust_markowitz = trust_markowitz
        self.gamma = self.assign_gamma(target_gamma)
        self.resample = resample
        self.type = type.lower()
        self.ticker = returns.columns if isinstance(returns, pd.DataFrame) else [returns.name]
        self.returns = returns
        self.expected_returns = self.get_expected_returns()
        self.expected_covariance = self.get_expected_covariance()
        self.dim = len(self.expected_returns)
        self.len = len(self.returns)

        # Initialize frontier
        self.frontier = pd.DataFrame()

        # Initialize actual_weights and actual_returns
        self.actual_weights = pd.DataFrame(columns=self.ticker)
        self.actual_returns = pd.Series(dtype=float)

        # Get optimal weights
        self.optimal_weights = self.get_optimal()
        print(f"Optimal Weights for {self.type}: {self.optimal_weights}")

        # Calculate portfolio metrics
        self.expected_portfolio_return = self.get_expected_portfolio_return()
        self.expected_portfolio_varcov = self.get_expected_portfolio_varcov()

        if self.type != 'erc':
            Portfolio.non_combined_portfolios.append(self)
        if self.type == 'erc':
            if not self.frontier.empty:
                self.frontier.loc[0, 'expected_return'] = self.expected_portfolio_return
                self.frontier.loc[0, 'expected_variance'] = self.expected_portfolio_varcov
                if self.expected_portfolio_varcov > 0:
                    self.frontier.loc[0, 'expected_sharpe'] = self.expected_portfolio_return / np.sqrt(
                        self.expected_portfolio_varcov)
                else:
                    self.frontier.loc[0, 'expected_sharpe'] = 0
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
        expected_variances_vector = np.einsum('ij,jk,ik->i', frontier_weights, self.expected_covariance,
                                              frontier_weights)

        expected_sharpe = np.zeros_like(expected_returns_vector)
        non_zero_variance = expected_variances_vector > 0
        expected_sharpe[non_zero_variance] = (
                expected_returns_vector[non_zero_variance] / np.sqrt(expected_variances_vector[non_zero_variance]))
        expected_sharpe[~non_zero_variance] = 0  # Handle zero variance cases

        data = {
            'gamma': self.gamma_linspace if not singular else [singular],
            'expected_return': expected_returns_vector,
            'expected_variance': expected_variances_vector,
            'expected_sharpe': expected_sharpe}

        # Add weights for each asset
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
            iterator = enumerate(self.gamma_linspace)
        else:
            iterator = [(0, singular)]

        for i, gamma in iterator:
            def objective(weights):
                return 0.5 * np.dot(weights.T, np.dot(self.expected_covariance, weights)) - gamma * np.dot(
                    self.expected_returns, weights)

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
                if result.success and np.all(result.x >= 0) and np.isfinite(result.x).all():
                    results[i, :] = result.x
                    initial_guess = result.x
                else:
                    print(f"Warning: Optimization failed for gamma {gamma}. Assigning equal weights.")
                    results[i, :] = np.ones(self.dim) / self.dim
            else:
                if result.success and np.all(result.x >= 0) and np.isfinite(result.x).all():
                    results = result.x
                else:
                    print(f"Warning: Optimization failed for singular gamma {gamma}. Assigning equal weights.")
                    results = np.ones(self.dim) / self.dim
        return results

    def _efficient_frontier_cvxpy(self, singular=None):
        weights = cp.Variable(self.dim)
        gamma_param = cp.Parameter(nonneg=False)
        markowitz = 0.5 * cp.quad_form(weights, cp.psd_wrap(
            self.expected_covariance)) - gamma_param * self.expected_returns.T @ weights
        constraints = [cp.sum(weights) == 1, weights >= 0]
        problem = cp.Problem(cp.Minimize(markowitz), constraints)

        if not singular:
            results = np.zeros((len(self.gamma_linspace), self.dim))
            iterator = enumerate(self.gamma_linspace)
        else:
            iterator = [(0, singular)]

        for i, gamma in iterator:
            gamma_param.value = gamma
            problem.solve(warm_start=True)
            if not singular:
                if weights.value is not None and np.all(weights.value >= 0) and np.isfinite(weights.value).all():
                    results[i, :] = weights.value
                else:
                    print(f"Warning: Optimization failed for gamma {gamma}. Assigning equal weights.")
                    results[i, :] = np.ones(self.dim) / self.dim
            else:
                if weights.value is not None and np.all(weights.value >= 0) and np.isfinite(weights.value).all():
                    results = weights.value
                else:
                    print(f"Warning: Optimization failed for singular gamma {gamma}. Assigning equal weights.")
                    results = np.ones(self.dim) / self.dim
        return results

    def get_expected_returns(self) -> pd.DataFrame | pd.Series:
        # TODO: Attention! If extending beyond ERC, if statement must be updated.
        if self.trust_markowitz and self.type == 'erc':
            internal_expectations = np.array(
                [portfolio.expected_portfolio_return for portfolio in Portfolio.non_combined_portfolios])
            return pd.Series(internal_expectations, index=self.returns.columns)
        return self.returns.mean(axis=0)

    def get_expected_covariance(self) -> pd.DataFrame | pd.Series:
        if self.trust_markowitz and self.type == 'erc':
            internal_expectations = np.array(
                [np.sqrt(portfolio.expected_portfolio_varcov) for portfolio in Portfolio.non_combined_portfolios])
            sample_correlations = self.returns.corr().fillna(0)
            varcov_matrix = np.outer(internal_expectations, internal_expectations) * sample_correlations
            return pd.DataFrame(varcov_matrix, index=self.returns.columns, columns=self.returns.columns)
        return self.returns.cov(ddof=0)

    def get_expected_portfolio_return(self) -> float:
        return np.dot(self.expected_returns, self.optimal_weights)

    def get_expected_portfolio_varcov(self) -> float:
        return self.optimal_weights.T @ self.expected_covariance @ self.optimal_weights

    def _fit_erc(self):
        weights = cp.Variable(self.dim)
        objective = cp.Minimize(0.5 * cp.quad_form(weights, self.expected_covariance))

        # Avoid log(0) by setting a lower bound
        epsilon = 1e-8
        log_constraint = cp.sum(cp.log(weights + epsilon)) >= np.log(epsilon) * self.dim - 2  # Adjusted constraint
        constraints = [weights >= epsilon, weights <= 1, cp.sum(weights) == 1, log_constraint]

        problem = cp.Problem(objective, constraints)
        problem.solve(warm_start=True)

        if weights.value is None:
            result = self._fit_erc_robust()
        else:
            total_weight = np.sum(weights.value)
            if total_weight > 0:
                result = weights.value / total_weight
            else:
                print("Warning: Total weight is zero in ERC optimization.")
                result = np.zeros(self.dim)
        return result

    def _fit_erc_robust(self) -> np.ndarray:
        print("ERC optimization failed to find a solution. Attempting robust optimization.")

        def _ERC(x, cov_matrix):
            volatility = np.sqrt(x.T @ cov_matrix @ x)
            if volatility == 0:
                return np.inf
            abs_risk_contribution = x * (cov_matrix @ x) / volatility
            mean = np.mean(abs_risk_contribution)
            return np.sum((abs_risk_contribution - mean) ** 2)

        bounds = Bounds(1e-8, 1)  # Avoid zero weights
        lc = LinearConstraint(np.ones(self.dim), 1, 1)
        settings = {'tol': 1e-16, 'method': 'SLSQP'}
        res = minimize(_ERC, np.full(self.dim, 1 / self.dim), args=(self.expected_covariance), constraints=[lc],
                       bounds=bounds, **settings)
        if res.success:
            return res.x
        else:
            print("Robust ERC optimization failed.")
            return np.ones(self.dim) / self.dim  # Assign equal weights if optimization fails

    def evaluate_performance(self, evaluationData: pd.DataFrame | pd.Series) -> pd.Series:
        # Initialize actual_weights and actual_returns
        self.actual_weights = pd.DataFrame(columns=self.ticker)
        self.actual_returns = pd.Series(dtype=float, index=evaluationData.index if evaluationData is not None else [])

        if evaluationData.empty or evaluationData.isna().all().all() or (evaluationData == 0).all().all():
            print("No data available for evaluation.")
            return self.actual_returns  # Return the initialized empty Series

        portfolioWeights = self.optimal_weights
        subperiodReturns = []
        subperiodWeights = [portfolioWeights]
        for idx, singleSubperiodReturns in zip(evaluationData.index, evaluationData.values):
            portfolioReturns = subperiodWeights[-1] @ singleSubperiodReturns
            # Prevent division by zero
            if (1 + portfolioReturns) == 0:
                portfolioReturns = 0
            portfolioWeights = subperiodWeights[-1] * (1 + singleSubperiodReturns) / (1 + portfolioReturns)
            subperiodReturns.append(portfolioReturns)
            subperiodWeights.append(portfolioWeights)

        self.actual_returns = pd.Series(subperiodReturns, index=evaluationData.index)
        self.actual_weights = pd.DataFrame(subperiodWeights[:-1], index=evaluationData.index, columns=self.ticker)

        print(f"Actual Returns for Portfolio: {self.actual_returns}")
        print(f"Actual Weights for Portfolio: {self.actual_weights.head()}")  # Print first few weights

        return self.actual_returns

# Function to Optimize and Evaluate All Portfolios for a Step
def optimize_and_evaluate_all(step, optimizationIndex, evaluationIndex):
    try:
        # Extract sample and evaluation data
        sampleEquity = equity_returns.loc[optimizationIndex]
        sampleMarketValues = market_values.loc[optimizationIndex]
        sampleMetals = {key: df.loc[optimizationIndex] for key, df in aligned_returns.items()}
        evaluationEquity = equity_returns.loc[evaluationIndex]
        evaluationMetals = {key: df.loc[evaluationIndex] for key, df in aligned_returns.items()}

        # Apply scoring filters
        minMarketCapThreshold = 0
        maxMarketCapThreshold = np.inf
        nullFilter = create_filter_mask_scoring(sampleEquity, sampleMarketValues,
                                                minMarketCapThreshold, maxMarketCapThreshold,
                                                price_threshold=0.03,  # Adjusted threshold
                                                return_threshold_high=0.15,  # Adjusted threshold
                                                return_threshold_low=-0.15,  # Adjusted threshold
                                                max_filter_failures=3)  # Adjusted max failures
        sampleEquity = sampleEquity.drop(columns=nullFilter, level=1)
        evaluationEquity = evaluationEquity.drop(columns=nullFilter, level=1)

        print(f"\nStep {step}: {len(sampleEquity.columns)} assets after filtering.")

        if sampleEquity.empty:
            print(f"Warning: No assets left after filtering for step {step}.")
            return {}

        # Portfolio definitions
        required_regions = ['AMER', 'EM', 'EUR', 'PAC']
        portfolio_definitions = []
        for region in required_regions:
            # Extract the relevant data for each region
            if region in sampleEquity.columns.levels[0]:
                sample_data = sampleEquity.xs(region, level=0, axis=1)
                evaluation_data = evaluationEquity.xs(region, level=0, axis=1)
                if not sample_data.empty and not evaluation_data.empty:
                    portfolio_definitions.append(
                        (f'equity_{region.lower()}', sample_data, evaluation_data))
                else:
                    print(f"Warning: Skipping portfolio for region {region} due to missing data.")
            else:
                print(f"Warning: Skipping portfolio for region {region} due to missing data.")

        # Append other portfolios (Metals, Commodities, Crypto, Volatilities) with checks
        for portfolio_name in ['Metals', 'Commodities', 'Crypto', 'Volatilities']:
            sample_data = sampleMetals.get(portfolio_name)
            evaluation_data = evaluationMetals.get(portfolio_name)
            if sample_data is not None and not sample_data.empty:
                portfolio_definitions.append((portfolio_name, sample_data, evaluation_data))
            else:
                print(f"Warning: Skipping portfolio {portfolio_name} due to missing or empty data.")

        results = {}
        for portfolio_name, sample_data, evaluation_data in portfolio_definitions:
            print(f"Processing portfolio: {portfolio_name}")
            if sample_data is not None and not sample_data.empty:
                portfolio = Portfolio(sample_data, 'max_sharpe')
                # Evaluate on optimization data (optional)
                optimization_performance = portfolio.evaluate_performance(sample_data)
                # Evaluate on evaluation data
                if evaluation_data is not None and not evaluation_data.empty:
                    evaluation_performance = portfolio.evaluate_performance(evaluation_data)
                else:
                    print(f"No evaluation data for {portfolio_name} at step {step}")
                    evaluation_performance = pd.Series(0,
                                                       index=evaluation_data.index if evaluation_data is not None else [])
                # Collect actual weights and returns
                results[portfolio_name] = {
                    'optimization': optimization_performance,
                    'evaluation': evaluation_performance,
                    'portfolio': portfolio,
                    'actual_weights': portfolio.actual_weights.copy(),
                    'actual_returns': portfolio.actual_returns.copy(),
                }
            else:
                print(f"No sample data for {portfolio_name} at step {step}")
        return results
    except Exception as e:
        print(f"Error in optimize_and_evaluate_all for step {step}: {e}")
        return {}

# Initialize Spinner
spinner = Spinner("Starting...")
spinner.start()
spinner.message("Loading data...", "blue")

# Load Data
staticData = pd.read_excel(staticPath, engine='openpyxl')
masterData = pd.read_excel(ritPath, usecols=lambda x: x != 'NAME', index_col=0, engine='openpyxl').transpose()
masterData.index.rename('DATE', inplace=True)
masterData = masterData[masterData.index.year > 2000]

capData = pd.read_excel(mvPath, usecols=lambda x: x != 'NAME', index_col=0, engine='openpyxl').transpose()
capData.index = pd.to_datetime(capData.index, format='%Y-%m-%d')
capData.index.rename('DATE', inplace=True)
capData = capData[capData.index.year > 2000] * 1e6

# Initialize Global Variables
masterIndex = masterData.index
global_tickers = list(masterData.columns)
df_dict = {}
mv_dict = {}

# Construct Equity Returns and Market Values by Region
for region in ['AMER', 'EM', 'EUR', 'PAC']:
    isin_filter = staticData['ISIN'][staticData['Region'] == region]
    if isin_filter.empty:
        print(f"Warning: No ISINs found for region {region}.")
        df_dict[region] = pd.DataFrame()
        mv_dict[region] = pd.DataFrame()
    else:
        df_dict[region] = masterData[isin_filter].pct_change()  # returns for the region
        mv_dict[region] = capData[isin_filter]

print("\nEquity Returns by Region:")
print(df_dict)
equity_returns = pd.concat(df_dict, axis=1)
print("\nCombined Equity Returns:")
print(equity_returns)
market_values = pd.concat(mv_dict, axis=1)

# Add Tickers from Subportfolios
folders = ['Metals', 'Commodities', 'Volatilities', 'Crypto']
global_tickers = add_tickers(root, folders, global_tickers)

# Generate Aligned Returns for All Subportfolios
aligned_returns = generate_aligned_returns(root, folders, masterIndex)

print("\nAligned Returns:")
print(aligned_returns)

# Clean Aligned Returns and Equity Returns
for key, df in aligned_returns.items():
    df.replace([np.inf, -np.inf, -1], np.nan, inplace=True)
    df.fillna(0, inplace=True)

equity_returns.replace([np.inf, -np.inf, -1], np.nan, inplace=True)
equity_returns.fillna(0, inplace=True)

# Initialize Portfolio Returns DataFrame
portfolio_keys = ['equity_amer', 'equity_em', 'equity_pac', 'equity_eur', 'Metals', 'Commodities', 'Volatilities', 'Crypto']
portfolio_returns = pd.DataFrame(index=masterIndex, columns=[*portfolio_keys, 'ERC'])
portfolio_returns[:] = 0

# Define Iteration Depth Function
def iteration_depth(limit=None, frequency="annual"):
    if frequency == "annual":
        if limit is None:
            YYYY = 2021
        else:
            YYYY = limit
        indexIterator = {0: {'optimizationIndex': masterIndex.year < 2006, 'evaluationIndex': masterIndex.year == 2006}}
        for year, index in zip(range(2007, YYYY + 1), range(1, (YYYY - 2006) + 1)):
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

# Initialize Visual Data Dictionary
visual_data = {}
indexIterator = iteration_depth(limit=2022, frequency="annual")
print("\nIndex Iterator:")
print(indexIterator)
spinner.message('Optimizing', 'yellow')

# Start Timer for Execution
start_second = time.time()
all_results = {}

# Execute Optimization and Evaluation in Parallel
with ThreadPoolExecutor() as executor:
    futures = {
        executor.submit(optimize_and_evaluate_all, step, indexIterator[step]['optimizationIndex'],
                        indexIterator[step]['evaluationIndex']): step
        for step in indexIterator
    }
    for future in concurrent.futures.as_completed(futures):
        step = futures[future]
        try:
            result = future.result()
            all_results[step] = result
        except Exception as exc:
            print(f"Step {step} generated an exception: {exc}")

# Ensure that the portfolio_returns DataFrame has the correct index
portfolio_returns = portfolio_returns.reindex(masterIndex)

# Initialize dictionaries to collect weights and returns
all_actual_weights = {key: pd.DataFrame() for key in portfolio_keys + ['ERC']}
all_actual_returns = {key: pd.Series(dtype=float) for key in portfolio_keys + ['ERC']}

# Populate portfolio_returns and collect actual weights and returns
for step, result in all_results.items():
    optimizationIndex = indexIterator[step]['optimizationIndex']
    evaluationIndex = indexIterator[step]['evaluationIndex']

    print(f"\nProcessing Step {step}:")
    print(f"Optimization Index sum: {optimizationIndex.sum()}, Evaluation Index sum: {evaluationIndex.sum()}")

    for portfolio_name, data in result.items():
        if data and 'evaluation' in data and isinstance(data['evaluation'], pd.Series):
            if not data['evaluation'].isna().all():
                # Assign evaluation data using the indices from data['evaluation']
                portfolio_returns.loc[data['evaluation'].index, portfolio_name] = data['evaluation']
                print(f"Assigned evaluation data for {portfolio_name} at step {step}.")
                # Collect actual weights and returns
                if not data['actual_weights'].empty:
                    all_actual_weights[portfolio_name] = pd.concat(
                        [all_actual_weights[portfolio_name], data['actual_weights']], axis=0)
                if not data['actual_returns'].empty:
                    all_actual_returns[portfolio_name] = pd.concat(
                        [all_actual_returns[portfolio_name], data['actual_returns']], axis=0)
            else:
                print(f"Warning: Evaluation data for {portfolio_name} at step {step} contains only NaNs.")
        else:
            print(f"Warning: Missing or invalid evaluation data for {portfolio_name} at step {step}.")

    # Construct the ERC Portfolio
    samplePortfolio = portfolio_returns.loc[optimizationIndex, portfolio_keys]
    evaluationPortfolio = portfolio_returns.loc[evaluationIndex, portfolio_keys]

    if not samplePortfolio.empty and not evaluationPortfolio.empty:
        ercPortfolio = Portfolio(samplePortfolio, 'erc', trust_markowitz=False)
        erc_performance = ercPortfolio.evaluate_performance(evaluationPortfolio)
        if not erc_performance.isna().all():
            # Assign ERC performance data using the indices from erc_performance
            portfolio_returns.loc[erc_performance.index, 'ERC'] = erc_performance
            print(f"Assigned ERC performance data at step {step}.")
            # Collect actual weights and returns for ERC
            if not ercPortfolio.actual_weights.empty:
                all_actual_weights['ERC'] = pd.concat(
                    [all_actual_weights['ERC'], ercPortfolio.actual_weights], axis=0)
            if not ercPortfolio.actual_returns.empty:
                all_actual_returns['ERC'] = pd.concat(
                    [all_actual_returns['ERC'], ercPortfolio.actual_returns], axis=0)
        else:
            print(f"Warning: ERC performance data at step {step} contains only NaNs.")
    else:
        print(f"Warning: Empty portfolio data for ERC calculation at step {step}.")

print("\nFinal Portfolio Returns:")
print(portfolio_returns)
end_second = time.time()
print("\nFinished execution time: ", (end_second - start_second))

# Stop Spinner
spinner.stop()

# Collect and Display Actual Weights and Returns Over Time
for portfolio_name in portfolio_keys + ['ERC']:
    all_actual_weights[portfolio_name].sort_index(inplace=True)
    all_actual_returns[portfolio_name].sort_index(inplace=True)
    print(f"\nActual Weights over time for {portfolio_name}:")
    print(all_actual_weights[portfolio_name])
    print(f"\nActual Returns over time for {portfolio_name}:")
    print(all_actual_returns[portfolio_name])

# Save the actual weights and returns to CSV files if needed
# for portfolio_name in portfolio_keys + ['ERC']:
#     all_actual_weights[portfolio_name].to_csv(f"{portfolio_name}_weights.csv")
#     all_actual_returns[portfolio_name].to_csv(f"{portfolio_name}_returns.csv")
