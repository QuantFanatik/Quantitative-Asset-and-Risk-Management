import pandas as pd
import os

def pct_skip_missing(df, keys):
    df_filled = df[keys].ffill()
    pct_change = df_filled.pct_change()
    pct_change[df[keys].isna()] = None
    return pct_change

dir_path = os.path.dirname(os.path.realpath(__file__))
save_path = os.path.join(dir_path, "data/data_YF")

# -------------------------------------------
# Create a master dataframe with all the data
# -------------------------------------------

directories = ["Commodities", "Crypto", "Metals", "Volatilities"]
common_index = pd.bdate_range(start='1980-01-01', end='2030-01-01', freq='D', tz='UTC')
index_union = None
for directory in directories:
    path = os.path.join(save_path, directory)
    for filename in os.listdir(path):
        if filename == 'some_files':
            continue
        subpath = os.path.join(path, filename)
        df = pd.read_csv(subpath, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True).normalize()
        if index_union is None:
            index_union = df.index
        else:
            index_union = index_union.union(df.index)

master_df = pd.DataFrame(index=index_union)
for directory in directories:
    path = os.path.join(save_path, directory)
    for filename in os.listdir(path):
        if filename == 'some_files':
            continue
        name = filename.split(".")[0]
        subpath = os.path.join(path, filename)
        df = pd.read_csv(subpath, index_col=0)['Close']
        df.index = pd.to_datetime(df.index, utc=True).normalize()
        df = df.reindex(index_union)
        master_df[name] = df

# print(master_df.tail(10))
# master_df = pct_skip_missing(master_df, master_df.columns)
# print(master_df.tail(10))
master_df.to_csv(os.path.join(save_path, "master_df.csv"))

# -------------------------
# Read the master dataframe
# -------------------------

dir_path = os.path.dirname(os.path.realpath(__file__))
master_path = os.path.join(dir_path, "data/data_YF/master_df.csv")

master_df = pd.read_csv(master_path, index_col=0, parse_dates=True)
print(master_df.tail(10))

