
from sklearn.metrics import mean_absolute_error, precision_score

def evaluate_model(actual, predicted):
    return mean_absolute_error(actual, predicted)

def evaluate_precision(actual, predicted):
    return precision_score(actual, predicted)