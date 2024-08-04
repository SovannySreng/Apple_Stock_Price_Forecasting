
from src.data_preprocessing import load_data, preprocess_data
from src.eda import eda
from src.feature_engineering import feature_engineering
from src.model_training import train_model
from src.evaluation import evaluate_model
from src.visualization import plot_histograms, plot_time_series
from src.utils import setup_logging, log_error

def main():
    setup_logging()
    
    try:
        df = load_data('data/AAPL.csv')  # Use relative path
        
        # Perform EDA
        eda(df)
        
        # Preprocess Data
        df = preprocess_data(df)
        
        # Feature Engineering
        df = feature_engineering(df)
        
        # Visualizations
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        date_col = 'Date'
        value_col = 'Close'
        
        plot_histograms(df, num_cols)
        plot_time_series(df, date_col, value_col)
        
        # Train the model
        X = df.drop(columns=['Close', 'Date'])  # Adjust according to your feature columns
        y = df['Close']
        model, x_test, y_test = train_model(X, y)
        
        # Evaluate the model
        evaluate_model(model, x_test, y_test)
        
    except Exception as e:
        log_error(e)
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()