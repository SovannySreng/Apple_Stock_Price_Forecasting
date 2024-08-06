import pandas as pd
import src.eda as eda
import src.evaluation as eval
import src.feature_engineering as fe
import src.model_training as mt
import src.visualization as viz
from src.data_preprocessing import load_data, preprocess_data, split_data

def main():
    # Load and preprocess data
    data = load_data(ticker="AAPL", start_date="2000-01-01", end_date="2022-05-31")
    df = preprocess_data(data)

    # EDA
    eda.plot_line(df, 'Close')
    eda.decompose_series(df, 'Close')
    eda.plot_acf_pacf(df, 'Close')
    is_stationary = eda.check_stationarity(df, 'Close')
    print(f"Is the series stationary? {is_stationary}")

    # Feature Engineering
    data = fe.create_features(df)

    # Train ARIMA model
    arima_model = mt.train_arima_model(df['Close'], order=(1,1,1))
    forecast = arima_model.get_forecast(2)
    ypred = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=0.05)

    # Evaluate ARIMA model
    actual = pd.Series([184.40, 185.04], index=pd.to_datetime(['2024-01-01', '2024-02-01']))
    pred = pd.Series(ypred.values, index=pd.to_datetime(['2024-01-01', '2024-02-01']))
    mae = eval.evaluate_model(actual, pred)
    print(f'ARIMA Model MAE: {mae}')

    # Visualize ARIMA forecast
    forecast_df = pd.DataFrame({
        'price_actual': actual,
        'price_predicted': pred,
        'lower_int': conf_int['lower Close'],
        'upper_int': conf_int['upper Close']
    })
    viz.plot_forecast(df, forecast_df, conf_int)

    # Train XGBoost model
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    xgb_model = mt.train_xgboost_model(data.iloc[:-30], features)

    # Backtest XGBoost model
    predictions = fe.backtest(data, xgb_model, features)
    xgb_precision = eval.evaluate_precision(predictions['Target'], predictions['predictions'])
    print(f'XGBoost Model Precision: {xgb_precision}')

if __name__ == '__main__':
    main()