from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBClassifier

def train_arima_model(series, order):
    model = ARIMA(series, order=order)
    return model.fit()

def train_xgboost_model(train, features):
    model = XGBClassifier(max_depth=3, n_estimators=100, random_state=42)
    model.fit(train[features], train['Target'])
    return model