import pandas as pd
import yfinance as yf

def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

def preprocess_data(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data = data.asfreq('D')  # Set frequency to daily
    data['Close'] = data['Close'].interpolate()  # Interpolate to fill missing values if any
    data['Next_day'] = data['Close'].shift(-1)
    data['Target'] = (data['Next_day'] > data['Close']).astype(int)
    return data

def split_data(data, target_col, test_size=0.2, random_state=1234):
    from sklearn.model_selection import train_test_split
    x = data.drop(target_col, axis=1)
    y = data[target_col]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test