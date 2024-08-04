
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def eda(df: pd.DataFrame):
    print(df.head())
    print(df.info())
    print(df.describe())
    sns.pairplot(df)
    plt.show()