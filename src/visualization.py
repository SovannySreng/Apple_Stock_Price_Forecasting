
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_histograms(df: pd.DataFrame, num_cols: list):
    df[num_cols].hist(figsize=(14, 14))
    plt.show()

def plot_time_series(df: pd.DataFrame, date_col: str, value_col: str):
    plt.figure(figsize=(10, 5))
    plt.plot(df[date_col], df[value_col])
    plt.title(f'Time Series of {value_col}')
    plt.xlabel('Date')
    plt.ylabel(value_col)
    plt.show()