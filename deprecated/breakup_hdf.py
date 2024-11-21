import pandas as pd

# Load the original large HDF5 file
file_path = 'data/efficient_frontiers.hdf'
data = pd.read_hdf(file_path)

# Split data by year and ensure `year` is preserved
years = data.index.get_level_values("year").unique()
for year in years:
    yearly_data = data.xs(year, level="year")
    
    # Add `year` back as a column if it’s not part of the index in each chunk
    yearly_data = yearly_data.reset_index().assign(year=year).set_index(['year'] + yearly_data.index.names)
    
    # Save as a separate HDF5 file
    yearly_data.to_hdf(f'data/efficient_frontiers_{year}.hdf', key='frontier_data')