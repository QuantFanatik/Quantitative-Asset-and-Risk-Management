# Portfolio Optimization Web Application

## Overview

This project is a **Portfolio Optimization Web Application** developed as part of our **QARM (Quantitative Asset and Risk Management)** course. It represents a collaborative group effort to construct an interactive tool that assists users in creating optimal investment portfolios. The application is designed to process a dataset of over **2,100 assets** spanning various asset classes to explore, optimize, and evaluate portfolios.

The project combines theoretical concepts from quantitative finance, such as mean-variance optimization and risk profiling, with practical implementation using Python and Streamlit.

---

## Features

### 1. Risk Profiling
- **Gamma Calculation:** Users can input their risk aversion parameter (`Gamma`) directly or answer a series of questions to estimate it.
- **Dynamic Adjustment:** The application adjusts the `Gamma` to the nearest available value in the dataset for compatibility with optimization computations.

### 2. Data Exploration
- **Asset Classes:** Analyze different asset classes, including:
  - Metals
  - Commodities
  - Cryptocurrencies (e.g., Bitcoin, Ethereum)
  - Volatility indices (e.g., VIX, MOVE)
  - Regional equity markets (North America, Europe, Asia-Pacific, and Emerging Markets)
- **Visualizations:**
  - Annualized returns and volatilities.
  - Correlation heatmaps for intra-class relationships.
  - Efficient frontiers with risk-return trade-offs.

### 3. Sub-Portfolio Optimization
- Individual optimization of sub-portfolios for each asset class using **Mean-Variance Optimization (MVO)**.
- Supports dynamic weight adjustments for assets based on historical data.
- Visualizes weight allocations over time and on specific dates.

### 4. Global Portfolio Optimization
- **Equal Risk Contribution (ERC) Portfolio:** Combines optimized sub-portfolios into a globally diversified portfolio.
- **Dynamic Rebalancing:** Adjusts weights periodically to maintain risk allocation targets.

### 5. Performance Analysis
- **Cumulative Returns and Drawdowns:** Evaluate the performance of selected portfolios over time.
- **Metrics:** Provides key performance indicators such as:
  - Mean annualized return.
  - Annualized volatility.
  - Sharpe ratio.
  - Maximum drawdown and drawdown duration.
- **Correlation Analysis:** Heatmaps for understanding relationships between selected portfolios.

---

## Technical Structure

### 1. Data Handling
- **Dataset:** The application processes data for over 2,100 assets from multiple asset classes.
- **File Structure:** Data is split into chunks (e.g., CSV files) for efficient processing and is dynamically loaded using utility functions.
- **Preprocessing:** Includes handling missing values, normalizing weights, and calculating portfolio returns and volatilities.

### 2. Core Modules
- **Risk Profiling:**
  - Gamma estimation through user input or questionnaire.
  - Normalization and nearest value adjustments.
- **Efficient Frontier Calculations:**
  - Calculates expected returns, variances, and Sharpe ratios for different portfolios.
  - Visualizes efficient frontiers and capital market lines.
- **Dynamic Weight Adjustments:**
  - Simulates rebalancing based on monthly or yearly intervals.
  - Ensures normalized weights for portfolio construction.

### 3. Interactive Visualizations
- Utilizes libraries like **Plotly**, **Matplotlib**, and **Seaborn** for high-quality charts.
- Provides interactivity through Streamlit widgets like sliders, radio buttons, and dropdowns.

### 4. Performance and Metrics
- Tracks and visualizes cumulative returns and drawdowns for user-selected portfolios.
- Computes advanced metrics such as Sharpe ratio and drawdown durations.

---

## Acknowledgments

We would like to express our sincere gratitude to the developers and contributors of the libraries used in this project, including **pandas**, **numpy**, **matplotlib**, **seaborn**, **plotly**, and **Streamlit**. Your incredible work has made it possible for us to bring this application to life.

A special thank you to **Professor Divernois** for designing such an engaging and intellectually stimulating project. Your guidance and teaching have provided us with invaluable insights into quantitative finance and risk management, making this learning experience both challenging and rewarding.

Thank you for inspiring us to apply theory to practice in such an impactful way!


