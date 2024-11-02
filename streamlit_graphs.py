import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import os

# Streamlit title
st.title('Efficient Frontiers Across Portfolios with Year Slider')

# Path to the HDF5 file
root = os.path.dirname(__file__)
file_path = os.path.join(root, 'data', 'efficient_frontiers.hdf')

# Load the HDF file with caching, merge the separate files
@st.cache_data
def load_data():
    files = [f for f in os.listdir('data') if f.startswith("efficient_frontiers_") and f.endswith(".hdf")]
    data_chunks = []
    
    for file in files:
        chunk = pd.read_hdf(os.path.join('data', file))
        
        # Ensure `year` is part of the index after loading
        if 'year' not in chunk.index.names:
            chunk = chunk.set_index(['year'], append=True)
        
        data_chunks.append(chunk)
    
    # Concatenate all chunks into a single DataFrame
    merged_data = pd.concat(data_chunks).sort_index()
    return merged_data

# Generate efficient frontiers with Plotly slider for year control
@st.cache_data
def plot_efficient_frontiers_with_slider(data):
    years = data.index.get_level_values("year").unique()
    portfolios = data.index.get_level_values("portfolio").unique()

    # Initialize Plotly figure
    fig = go.Figure()

    # Create a trace for each portfolio's efficient frontier for each year
    for year in years:
        yearly_data = data.xs(year, level="year")
        
        for portfolio in portfolios:
            if portfolio not in yearly_data.index.get_level_values("portfolio").unique():
                continue  # Skip if the portfolio is not present in the current year
            
            portfolio_data = yearly_data.xs(portfolio, level="portfolio")
            x = portfolio_data[("metrics", "expected_variance")]
            y = portfolio_data[("metrics", "expected_return")]

            # Add trace for the portfolio's efficient frontier in this year
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines+markers',
                name=f"{portfolio} ({year})",
                visible=(year == years[0]),  # Only the first year is visible initially
                legendgroup=str(year),  # Group traces by year for easier toggling
                hovertemplate=f"<b>Portfolio:</b> {portfolio}<br><b>Year:</b> {year}<br>Expected Return: %{{y}}<br>Variance: %{{x}}"
            ))

    # Define slider steps for each year
    steps = []
    for i, year in enumerate(years):
        step = dict(
            method="restyle",
            args=["visible", [False] * len(fig.data)],  # Start by hiding all traces
            label=str(year),
        )
        
        # Make only traces for the current year visible
        for j in range(i * len(portfolios), (i + 1) * len(portfolios)):
            step["args"][1][j] = True  # Set visibility to True for current year's traces

        steps.append(step)

    # Add slider to the layout
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Year: "},
        pad={"t": 50},
        steps=steps
    )]

    # Update layout with title and slider
    fig.update_layout(
        title="Efficient Frontiers for All Portfolios Over Time",
        xaxis_title="Variance (Risk)",
        yaxis_title="Expected Return",
        sliders=sliders,
        xaxis_range=[0, 0.1],  # Limit the x-axis between 0 and 0.1
        yaxis_range=[-0.1, 0.2],  # Limit the y-axis between -0.1 and 0.2
        width=800,
        height=600
    )

    return fig

# Filter ERC portfolio data for a specific gamma
@st.cache_data
def get_erc_data_for_gamma(data, gamma):
    # Select ERC portfolio data and filter by gamma
    erc_data = data.xs('ERC', level='portfolio')
    gamma_data = erc_data.xs(gamma, level='gamma')

    # Drop any columns with NaN across all years
    gamma_data = gamma_data.dropna(axis=1, how='all')

    # Select only the weight columns
    weight_data = gamma_data.loc[:, "weights"]
    
    return weight_data

# Generate stacked area chart for ERC portfolio weights over time with a Plotly slider for gamma
@st.cache_data
def plot_erc_composition(data):
    gammas = data.index.get_level_values("gamma").unique()
    years = data.index.get_level_values("year").unique()

    # Initialize figure
    fig = go.Figure()

    # Add a trace for each asset's weights over time, for each gamma value
    for gamma in gammas:
        weight_data = get_erc_data_for_gamma(data, gamma)
        
        # Create stacked area traces for each asset in the ERC portfolio for this gamma
        for asset in weight_data.columns:
            fig.add_trace(go.Scatter(
                x=years,
                y=weight_data[asset],
                mode='lines',
                stackgroup='one',  # Creates a stacked area chart
                name=asset,
                visible=(gamma == gammas[0])  # Show only the first gamma initially
            ))

    # Define slider steps for gamma values
    steps = []
    for i, gamma in enumerate(gammas):
        step = dict(
            method="restyle",
            args=["visible", [False] * len(fig.data)],  # Start by hiding all traces
            label=f"{gamma:.2f}",  # Format gamma with two decimal places
        )

        # Make only the traces for the current gamma visible
        for j in range(i * len(weight_data.columns), (i + 1) * len(weight_data.columns)):
            step["args"][1][j] = True

        steps.append(step)

    # Define slider with formatted labels
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Gamma: ", "font": {"size": 16}},
        pad={"t": 50},
        steps=steps
    )]

    # Update layout
    fig.update_layout(
        title="ERC Portfolio Composition Over Time by Gamma",
        xaxis_title="Year",
        yaxis_title="Portfolio Weight",
        sliders=sliders,
        width=800,
        height=600
    )

    return fig

# Main logic
# Load data
data = load_data()

# Generate the efficient frontier plot with a Plotly slider
fig = plot_efficient_frontiers_with_slider(data)
st.plotly_chart(fig)

fig2 = plot_erc_composition(data)
st.plotly_chart(fig2)