import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds
import cvxpy as cp
import time

def generate_synthetic_data(N, annualize=252):
    np.random.seed(0)
    expected_returns = np.random.uniform(-0.15, 0.45, N)
    A = np.random.randn(N, N)
    sample_covariance = np.dot(A, A.T)
    return expected_returns, sample_covariance

def efficient_frontier_scipy(gammas, expected_returns, sample_covariance):
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

        result = minimize(objective,
                          initial_guess,
                          jac=jacobian,
                          constraints=constraints,
                          bounds=bounds,
                          method='SLSQP')
        
        optimized_weights = result.x
        results.append(optimized_weights)
        initial_guess = optimized_weights  
    return results

def efficient_frontier_cvxpy(gammas, expected_returns, sample_covariance):
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

def benchmark_crossover(gammas, max_dimension=100, step=1):
    results = []
    for N in range(1, max_dimension + 1, step):
        expected_returns, sample_covariance = generate_synthetic_data(N)

        start_time = time.perf_counter()
        efficient_frontier_scipy(gammas, expected_returns, sample_covariance)
        scipy_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        efficient_frontier_cvxpy(gammas, expected_returns, sample_covariance)
        cvxpy_time = time.perf_counter() - start_time

        results.append((N, scipy_time, cvxpy_time))
        print(f"N={N}, Scipy Time={scipy_time:.4f}s, CVXPY Time={cvxpy_time:.4f}s")

        if scipy_time > cvxpy_time:
            print(f"Crossover dimension found at N={N}")
            break

    return results

def efficient_frontier_adaptive(gammas, expected_returns, sample_covariance):
    if sample_covariance.shape[0] >= 27:
        return efficient_frontier_cvxpy(gammas, expected_returns, sample_covariance)
    else:
        results = efficient_frontier_scipy(gammas, expected_returns, sample_covariance)
        return results

gammas = np.linspace(-0.5, 1.5, 101)
results = benchmark_crossover(gammas)