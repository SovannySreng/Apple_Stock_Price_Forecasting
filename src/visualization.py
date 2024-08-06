import matplotlib.pyplot as plt

def plot_forecast(data, dp, conf_int):
    plt.figure(figsize=(10, 6))
    plt.plot(data['Close'], label='Actual')
    plt.plot(dp['price_predicted'], label='Predicted', color='orange')
    plt.fill_between(dp.index, dp['lower_int'], dp['upper_int'], color='k', alpha=0.1)
    plt.title('Actual vs Predicted Prices')
    plt.legend()
    plt.show()