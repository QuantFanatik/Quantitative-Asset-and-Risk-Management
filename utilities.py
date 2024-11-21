import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, minimize
import scipy.sparse.linalg as sparla
import cvxpy as cp
import os
import sys
import threading
import time
from itertools import cycle

class Settings:
    VALID_FREQUENCIES = ('monthly', 'annual')
    VALID_MODES = ('fast', 'gamma')

    def __init__(self):
        self.limit_year = None
        self.data_frequency = 'monthly'
        self.rebalancing_frequency = 'annual'
        self.ANNUALIZATION_FACTOR = 12
        self.master_index = None
        self.global_tickers = None
        self.mode = 'fast'
        self.gamma_linspace = np.linspace(-0.5, 1.5, 101)

        self.validate()

    def validate(self):
        if self.data_frequency not in self.VALID_FREQUENCIES:
            raise ValueError(f"Invalid data frequency: {self.data_frequency}. Must be one of {self.VALID_FREQUENCIES}.")
        if self.rebalancing_frequency not in self.VALID_FREQUENCIES:
            raise ValueError(f"Invalid rebalancing frequency: {self.rebalancing_frequency}. Must be one of {self.VALID_FREQUENCIES}.")
        if self.mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode: {self.mode}. Must be one of {self.VALID_MODES}.")

    def update_settings(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.validate()

settings = Settings()

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
    return (1 + sample_mean) ** settings.ANNUALIZATION_FACTOR - 1

def annualized_volatility(sample_std: float) -> float:
    return sample_std * np.sqrt(settings.ANNUALIZATION_FACTOR)

def sharpe_ratio(mean: float, volatility: float) -> float:
    if isinstance(volatility, pd.Series) and volatility.eq(0).any():
        return 0
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

def create_filter_mask1(sampleData, marketValuesData, minMarketCap: float = -np.inf, maxMarketCap: float = np.inf):
    # Get the latest date in the sample data
    latestDateSample = sampleData.index.max()

    # Filtering based on December returns
    decemberData = sampleData.loc[[latestDateSample]]
    decemberFilter = decemberData.columns[decemberData.iloc[0] == 0]

    # Price threshold filter for December (modify threshold as needed)
    yearEndPrices = sampleData.loc[latestDateSample]
    priceFilter = yearEndPrices[yearEndPrices < -np.inf].index  # Adjust threshold

    # High and low return filters
    returnFilterHigh = sampleData.columns[sampleData.max() >= 3]  # Adjust threshold
    returnFilterLow = sampleData.columns[sampleData.min() <= -3]  # Adjust threshold
    returnFilter = returnFilterHigh.union(returnFilterLow)

    # Frequent zero returns filter
    startOfYear = pd.Timestamp(latestDateSample.year, 1, 1)
    yearlyData = sampleData.loc[startOfYear:latestDateSample]
    monthsWithZeroReturns = (yearlyData == 0).sum(axis=0)
    frequentZerosFilter = monthsWithZeroReturns[monthsWithZeroReturns >= 11].index

    startOfYear = pd.Timestamp(latestDateSample.year-1, 1, 1)
    yearlyData = sampleData.loc[startOfYear:latestDateSample]
    monthsWithZeroReturns = (yearlyData == 0).sum(axis=0)
    frequentZerosFilter2 = monthsWithZeroReturns[monthsWithZeroReturns >= 15].index

    # Market cap filtering based on the latest date's data in marketValuesData
    marketValuesAtEnd = marketValuesData.loc[latestDateSample]
    marketCapFilterMin = marketValuesAtEnd[marketValuesAtEnd < minMarketCap].index
    marketCapFilterMax = marketValuesAtEnd[marketValuesAtEnd > maxMarketCap].index

    # Combine all filters, respecting the multi-index structure
    combinedFilter = decemberFilter.union(frequentZerosFilter).union(priceFilter).union(returnFilter)
    combinedFilter = combinedFilter.union(marketCapFilterMin).union(marketCapFilterMax).union(frequentZerosFilter2)

    # Return combined filter, maintaining multi-index compatibility
    return combinedFilter

class Portfolio:
    valid_types = ('markowitz', 'erc', 'max_sharpe', 'min_var')
    non_combined_portfolios = []
    
    def __new__(cls, *args, **kwargs):
        if settings.mode == 'fast':
            return FastPortfolio(*args, **kwargs)
        elif settings.mode == 'gamma':
            return GammaPortfolio(*args, **kwargs)
        else:
            raise ValueError(f"Invalid mode: {settings.mode}")

    
class FastPortfolio():
    valid_types = ('markowitz', 'erc', 'erc', 'max_sharpe', 'min_var')
    non_combined_portfolios = []

    def __init__(self, returns: pd.DataFrame | pd.Series, type: str='markowitz', names: list[str]=None, trust_markowitz: bool=False, resample: bool=False, main: bool=False):
        assert type.lower() in self.valid_types, f"Invalid type: {type}. Valid types are: {self.valid_types}"
        assert main or not trust_markowitz, "Non-main portfolios cannot trust Markowitz."
        if returns.isna().all().all() and not trust_markowitz:
            print("ERC sample is empty. Falling back to ex-ante expectations.")
            self.trust_markowitz = True
        else:
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

        if self.type != 'erc' or not main:
            Portfolio.non_combined_portfolios.append(self)


    def get_expected_returns(self) -> pd.DataFrame | pd.Series:
        #TODO: Attention! If extending beyond ERC, if statement must be updated.
        if self.trust_markowitz and self.type == 'erc':
            internal_expectations = np.array([portfolio.expected_portfolio_return for portfolio in Portfolio.non_combined_portfolios])
            return pd.Series(internal_expectations, index=self.returns.columns)
        elif self.type == 'erc' and self.returns.eq(0).all().all():
            return self.returns.mean(axis=0)
        return self.returns.mean(axis=0)
    
    def get_expected_covariance(self) -> pd.DataFrame | pd.Series:
        if self.trust_markowitz and self.type == 'erc':
            internal_expectations = np.array([np.sqrt(portfolio.expected_portfolio_varcov) for portfolio in Portfolio.non_combined_portfolios])
            sample_correlations = self.returns.corr().fillna(0)
            varcov_matrix = np.outer(internal_expectations, internal_expectations) * sample_correlations
            varcov_matrix = pd.DataFrame(varcov_matrix, index=self.returns.columns, columns=self.returns.columns)
        else:
            varcov_matrix = self.returns.cov(ddof=0)

        # Poison the covariance matrix for invalid assets
        null_variance_assets = self.returns.var(axis=0).eq(0)
        null_variance_assets = null_variance_assets[null_variance_assets].index.tolist()
        varcov_matrix.loc[null_variance_assets, :] = 0
        varcov_matrix.loc[:, null_variance_assets] = 0
        varcov_matrix.loc[null_variance_assets, null_variance_assets] = 100 + 10*np.random.rand()

        return varcov_matrix
    
    # TODO: Check annualization factor
    def get_expected_portfolio_return(self) -> float:
        return np.dot(self.expected_returns, self.optimal_weights) * settings.ANNUALIZATION_FACTOR
    
    def get_expected_portfolio_varcov(self) -> float:
        return self.optimal_weights.T @ self.expected_covariance @ self.optimal_weights * settings.ANNUALIZATION_FACTOR ** 2

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
            # TODO: verify necessity of annualization factor
            self.expected_returns = self._pandify(np.mean(simulated_returns, axis=0))
            self.expected_covariance = self._pandify(np.cov(simulated_returns.T, ddof=0))# * settings.ANNUALIZATION_FACTOR
            simulated_weights.append(method())
        
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
            # print("No data available for evaluation.")
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

        # print(self.expected_returns)
        # print(self.expected_covariance)
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
        if evaluationData.isna().all().all():
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

    @staticmethod
    def efficient_frontier(gammas, expected_returns, sample_covariance):
        if sample_covariance.shape[0] >= 30:
            return __class__._efficient_frontier_cvxpy(gammas, expected_returns, sample_covariance)
        else:
            results = __class__._efficient_frontier_scipy(gammas, expected_returns, sample_covariance)
            return results
        
    @staticmethod
    def _efficient_frontier_scipy(gammas, expected_returns, sample_covariance):
        dimension = sample_covariance.shape[0]
        initial_guess = np.ones(dimension) / dimension 
        constraints = [LinearConstraint(np.ones(dimension), 1, 1)]
        bounds = Bounds(0, 1)

        results = []
        for gamma in gammas:
            def objective(weights):
                return 0.5 * np.dot(weights.T, np.dot(sample_covariance, weights)) - gamma * np.dot(expected_returns, weights)
            def jacobian(weights):
                return np.dot(sample_covariance, weights) - gamma * expected_returns

            kwargs = {'fun': objective,
                    'jac': jacobian,
                    'x0': initial_guess,
                    'constraints': constraints,
                    'bounds': bounds,
                    'method': 'SLSQP',
                    'tol': 1e-16}
            result = minimize(**kwargs)
            
            optimized_weights = result.x
            results.append(optimized_weights)
            initial_guess = optimized_weights  
        return results

    @staticmethod
    def _efficient_frontier_cvxpy(gammas, expected_returns, sample_covariance):
        dimension = sample_covariance.shape[0]
        weights = cp.Variable(dimension)
        gamma_param = cp.Parameter(nonneg=False)
        markowitz = 0.5 * cp.quad_form(weights, sample_covariance) - gamma_param * expected_returns.T @ weights
        constraints = [cp.sum(weights) == 1, weights >= 0]
        problem = cp.Problem(cp.Minimize(markowitz), constraints)

        results = []
        for gamma_value in gammas:
            gamma_param.value = gamma_value
            problem.solve(warm_start=True)
            results.append(weights.value)
        return results
        
class GammaPortfolio():
    valid_types = ('markowitz', 'erc', 'max_sharpe', 'min_var')
    non_combined_portfolios = []
    # gamma_linspace = settings.gamma_linspace
    # print(gamma_linspace)
    
    def __init__(self, returns: pd.DataFrame | pd.Series, type: str='markowitz', names: list[str]=None, trust_markowitz: bool=False, resample: bool=False, target_gamma=None, main: bool=False):
        assert type.lower() in self.valid_types, f"Invalid type: {type}. Valid types are: {self.valid_types}"
        #TODO: Attention! ERC portfolios use sample returns, not ex-ante expectations.
        if returns.isna().all().all() and not trust_markowitz:
            print("ERC sample is empty. Falling back to ex-ante expectations.")
            self.trust_markowitz = True
        else:
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

        if self.type != 'erc' or not main:
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
        return min(settings.gamma_linspace, key=lambda x: abs(x - gamma))

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
            'gamma': settings.gamma_linspace if not singular else [singular],
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
            results = np.zeros((len(settings.gamma_linspace), self.dim))
            itterator = enumerate(settings.gamma_linspace)
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
            results = np.zeros((len(settings.gamma_linspace), self.dim))
            itterator = enumerate(settings.gamma_linspace)
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
            varcov_matrix = pd.DataFrame(varcov_matrix, index=self.returns.columns, columns=self.returns.columns)
        else:
            varcov_matrix = self.returns.cov(ddof=0)

        # Poison the covariance matrix for invalid assets
        null_variance_assets = self.returns.var(axis=0).eq(0)
        null_variance_assets = null_variance_assets[null_variance_assets].index.tolist()
        varcov_matrix.loc[null_variance_assets, :] = 0
        varcov_matrix.loc[:, null_variance_assets] = 0
        varcov_matrix.loc[null_variance_assets, null_variance_assets] = 100 + 10*np.random.rand()

        return varcov_matrix
    
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
        portfolioWeights = self.optimal_weights
        subperiodReturns = []
        subperiodWeights = [portfolioWeights]

        if evaluationData.isna().all().all() or (evaluationData == 0).all().all():
            print("No data available for evaluation.")
            self.actual_returns = pd.Series(0, index=evaluationData.index)
            self.actual_weights = pd.DataFrame(0, index=evaluationData.index, columns=self.ticker)
            return pd.Series(0, index=evaluationData.index)
        
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

def iteration_depth():
    limit = settings.limit_year
    frequency = settings.rebalancing_frequency
    masterIndex = settings.master_index
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


