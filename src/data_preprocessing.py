import os
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

#making a log directory(only 1 time)
log_dir = 'logs'
os.makedirs(log_dir, exist_ok= True)

logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def preprocess_df(df, is_train=True):
    """
    Preprocesses the DataFrame by standardizing the feature columns while preserving the target column.
    Returns DataFrame with standardized features and original target.
    """
    try:
        logger.debug('Starting preprocessing')
        
        # Separate features and target
        target = df['target']
        features = df.drop(columns=['target'])
        feature_columns = features.columns
        
        # Standardize features
        if is_train:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            logger.debug('Fitted new scaler on training data')
            
            # Save scaler
            os.makedirs('./models', exist_ok=True)
            joblib.dump(scaler, './models/scaler.pkl')
        else:
            scaler = joblib.load('./models/scaler.pkl')
            scaled_features = scaler.transform(features)
            logger.debug('Applied existing scaler to test data')
        
        # Recombine features and target
        processed_df = pd.DataFrame(scaled_features, columns=feature_columns)
        processed_df['target'] = target.values  # Preserve original target values
        
        logger.debug('Standardization complete')
        return processed_df
        
    except Exception as e:
        logger.error('Error during standardization: %s', e)
        raise

def main():
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        os.makedirs('./models', exist_ok=True)  # For scaler/model files
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded properly')

        # Transform the data
        logger.debug('Processing training data...')
        train_processed_data = preprocess_df(train_data, is_train=True)
        logger.debug('Processing test data...')
        test_processed_data = preprocess_df(test_data, is_train=False)

        # Store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        logger.debug('Processed data saved to %s', data_path)

    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()