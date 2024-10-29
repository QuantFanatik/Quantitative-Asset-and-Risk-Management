import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import os
import time

# Streamlit title
st.title('Interactive Equity Wealth Visualization')

st.cache_data.clear()

progress_text = "Operation in progress. Please wait."
my_bar = st.progress(0, text=progress_text)

# Ensure the relative path is handled correctly
root = os.path.dirname(__file__)  # Get the current script's directory
bonusPath = os.path.join(root, 'data', 'bonus.hdf')

# Load the HDF file with caching
@st.cache_data
def load_data():
    return pd.read_hdf(bonusPath)

# Cache the pivot table generation
@st.cache_data
def get_pivot_table(data):
    return data.reset_index().pivot(index='z_in', columns='z_stop', values='equity_wealth')

# Cache the 3D surface plot generation
@st.cache_data
def plot_3d_surface(pivot_table):
    X, Y = np.meshgrid(pivot_table.columns, pivot_table.index)
    Z = pivot_table.values
    
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig.update_layout(
        title='Equity Wealth 3D Surface Plot',
        scene=dict(
            xaxis_title='Z Stop',
            yaxis_title='Z In',
            zaxis_title='Equity Wealth'
        ),
        autosize=False,
        width=800,
        height=800,
        margin=dict(l=65, r=50, b=65, t=90)
    )
    
    return fig

# Cache the 2D slicing plot generation with slider
@st.cache_data
def plot_2d_slices(pivot_table):
    # Precompute all 2D slices for each z_in value
    slices = []
    z_in_values = pivot_table.index.values
    for z_in in z_in_values:
        slice_df = pivot_table.loc[z_in]
        slices.append(slice_df)

    # Create initial 2D plot with the first z_in value
    initial_slice_df = slices[0]
    slice_fig = go.Figure()

    # Add the first slice as the initial plot
    slice_fig.add_trace(go.Scatter(x=initial_slice_df.index, y=initial_slice_df.values, mode='lines+markers'))

    # Define the steps for the Plotly slider
    steps = []
    for i, z_in in enumerate(z_in_values):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(z_in_values)},  # Hide all traces initially
                  {"title": f"Equity Wealth vs Z Stop (Z In = {z_in})"}],  # Update title
            label=str(z_in)
        )
        step["args"][0]["visible"][i] = True  # Show only the current trace
        steps.append(step)

    # Add all slices to the figure, but only the first one is visible at the start
    for i, slice_df in enumerate(slices):
        slice_fig.add_trace(go.Scatter(x=slice_df.index, y=slice_df.values, mode='lines+markers', visible=(i == 0)))

    # Add slider to control the 2D slices
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Z In: "},
        pad={"t": 50},
        steps=steps
    )]

    # Update layout with slider
    slice_fig.update_layout(
        sliders=sliders,
        title=f"Equity Wealth vs Z Stop (Z In = {z_in_values[0]})",
        xaxis_title="Z Stop",
        yaxis_title="Equity Wealth",
        yaxis_range=[0, 20000],
        width=800,
        height=500
    )

    return slice_fig


# --- MAIN LOGIC ---

# Update progress bar: loading the data
my_bar.progress(20, text="Loading data...")
aggregator = load_data()

# Update progress bar: creating pivot table
my_bar.progress(50, text="Creating pivot table...")
pivot_table = get_pivot_table(aggregator)

# Update progress bar: generating 3D surface plot
my_bar.progress(70, text="Generating 3D surface plot...")
fig_3d = plot_3d_surface(pivot_table)

# Update progress bar: generating 2D plot with slider
my_bar.progress(90, text="Generating 2D plot with slider...")
fig_2d = plot_2d_slices(pivot_table)

# Final update of the progress bar
my_bar.progress(100, text="Completed!")

# Wait for a short time before revealing the plots
time.sleep(0.5)
my_bar.empty()  # Remove the progress bar once complete

# --- Reveal the figures after loading is done ---
st.plotly_chart(fig_3d)  # Display the 3D surface plots
st.plotly_chart(fig_2d)  # Display the 2D plot with the slider

# Terminal input to deploy the Streamlit app
# streamlit run Project/hdf_frontend_example.py