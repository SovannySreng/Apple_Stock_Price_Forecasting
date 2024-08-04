
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))