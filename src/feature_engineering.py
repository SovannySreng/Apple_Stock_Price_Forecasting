import pandas as pd

def create_features(df):
    df['Next_day'] = df['Close'].shift(-1)
    df['Target'] = (df['Next_day'] > df['Close']).astype(int)
    return df

def backtest(data, model, features, start=5031, step=120):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[:i].copy()
        test = data.iloc[i:(i+step)].copy()
        model_preds = predict(train, test, features, model)
        all_predictions.append(model_preds)

    return pd.concat(all_predictions)

def predict(train, test, features, model):
    model.fit(train[features], train['Target'])
    model_preds = model.predict(test[features])
    model_preds = pd.Series(model_preds, index=test.index, name='predictions')
    combine = pd.concat([test['Target'], model_preds], axis=1)
    return combine