
import logging
import os

def setup_logging():
    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    
    logging.basicConfig(filename=os.path.join(log_directory, 'app.log'), 
                        level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

def log_error(e):
    logging.error(f"Error: {e}")