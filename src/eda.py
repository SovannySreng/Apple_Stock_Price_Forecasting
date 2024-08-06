import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

def plot_line(df, column):
    plot = sns.lineplot(data=df, x=df.index, y=column)
    plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
    plt.show()

def decompose_series(df, column):
    decomposed = seasonal_decompose(df[column].dropna(), model='additive')
    trend = decomposed.trend
    seasonal = decomposed.seasonal
    residual = decomposed.resid
    plt.figure(figsize=(12,8))
    plt.subplot(411)
    plt.plot(df[column], label='Original', color='black')
    plt.legend(loc='upper left')
    plt.subplot(412)
    plt.plot(trend, label='Trend', color='red')
    plt.legend(loc='upper left')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonal', color='blue')
    plt.legend(loc='upper left')
    plt.subplot(414)
    plt.plot(residual, label='Residual', color='black')
    plt.legend(loc='upper left')
    plt.show()

def plot_acf_pacf(df, column):
    plt.rcParams.update({'figure.figsize':(7,4), 'figure.dpi':80})
    plot_acf(df[column].dropna())
    plot_pacf(df[column].dropna(), lags=11)
    plt.show()

def check_stationarity(df, column):
    result = adfuller(df[column].dropna())
    return result[1] < 0.05  # If p-value < 0.05, series is stationary