import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import joblib
import os

class SentimentPredictor:
    """Predicts stock price movements based on sentiment analysis"""
    
    def __init__(self, model_type='regression', algorithm='random_forest'):
        """
        Initialize predictor
        
        Parameters:
        - model_type: 'regression' for price prediction, 'classification' for direction prediction
        - algorithm: 'linear' or 'random_forest'
        """
        self.model_type = model_type
        self.algorithm = algorithm
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_features(self, sentiment_df, stock_df, sentiment_col='avg_sentiment', 
                         price_col='Close', date_col='date', window_size=5):
        """
        Prepare features for the prediction model
        
        Parameters:
        - sentiment_df: DataFrame with sentiment data
        - stock_df: DataFrame with stock price data
        - sentiment_col: Column name for sentiment score
        - price_col: Column name for stock price
        - date_col: Column name for date
        - window_size: Window size for rolling features
        
        Returns:
        - DataFrame with features and target variable
        """
        # Merge sentiment and stock data on date
        merged_df = pd.merge(stock_df, sentiment_df, on=date_col, how='inner')
        
        # Create lagged features
        for i in range(1, window_size + 1):
            # Lagged sentiment
            merged_df[f'sentiment_lag_{i}'] = merged_df[sentiment_col].shift(i)
            
            # Lagged price
            merged_df[f'price_lag_{i}'] = merged_df[price_col].shift(i)
            
            # Price percent change
            merged_df[f'price_pct_change_{i}'] = merged_df[price_col].pct_change(i)
            
        # Create rolling statistics for sentiment
        merged_df['sentiment_mean_3d'] = merged_df[sentiment_col].rolling(window=3).mean()
        merged_df['sentiment_std_3d'] = merged_df[sentiment_col].rolling(window=3).std()
        
        # Create target variable
        if self.model_type == 'regression':
            # Target: next day's price
            merged_df['target'] = merged_df[price_col].shift(-1)
        else:
            # Target: price direction (1 if price increases, 0 otherwise)
            merged_df['target'] = (merged_df[price_col].shift(-1) > merged_df[price_col]).astype(int)
            
        # Drop rows with NaN values
        merged_df = merged_df.dropna()
        
        return merged_df
        
    def select_features(self, data):
        """Select relevant features from the prepared data"""
        # Identify feature columns (those with 'sentiment', 'lag', 'pct_change', 'mean', 'std')
        feature_cols = [col for col in data.columns if any(
            keyword in col for keyword in ['sentiment', 'lag', 'pct_change', 'mean', 'std']
        )]
        
        # Split into features and target
        X = data[feature_cols]
        y = data['target']
        
        return X, y
        
    def train(self, sentiment_df, stock_df, test_size=0.2, random_state=42):
        """
        Train the prediction model
        
        Parameters:
        - sentiment_df: DataFrame with sentiment data
        - stock_df: DataFrame with stock price data
        - test_size: Proportion of data to use for testing
        - random_state: Random seed for reproducibility
        
        Returns:
        - Dictionary with model performance metrics
        """
        # Prepare data
        data = self.prepare_features(sentiment_df, stock_df)
        
        # Select features
        X, y = self.select_features(data)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train model
        if self.model_type == 'regression':
            if self.algorithm == 'linear':
                self.model = LinearRegression()
            else:  # random_forest
                self.model = RandomForestRegressor(n_estimators=100, random_state=random_state)
                
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Calculate correlation between predictions and actual values
            corr = np.corrcoef(y_test, y_pred)[0, 1]
            
            # Calculate direction accuracy (whether price movement direction was predicted correctly)
            direction_correct = np.sum((y_test.shift(1) < y_test) == (y_test.shift(1) < y_pred)) / len(y_test)
            
            # Return metrics
            return {
                'mse': mse,
                'rmse': rmse,
                'correlation': corr,
                'direction_accuracy': direction_correct
            }
            
        else:  # classification
            if self.algorithm == 'linear':
                self.model = LogisticRegression(random_state=random_state)
            else:  # random_forest
                self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
                
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Return metrics
            return {
                'accuracy': accuracy,
                'precision': report['1']['precision'],
                'recall': report['1']['recall'],
                'f1_score': report['1']['f1-score']
            }
            
    def predict_next_day(self, recent_sentiment, recent_prices):
        """
        Predict the next day's price or price movement
        
        Parameters:
        - recent_sentiment: DataFrame with recent sentiment data
        - recent_prices: DataFrame with recent price data
        
        Returns:
        - Predicted price or price movement direction
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
            
        # Prepare features for prediction
        data = self.prepare_features(recent_sentiment, recent_prices)
        
        # Select features
        X, _ = self.select_features(data)
        
        # Get the most recent data point
        latest_features = X.iloc[-1].values.reshape(1, -1)
        
        # Scale features
        latest_features_scaled = self.scaler.transform(latest_features)
        
        # Make prediction
        prediction = self.model.predict(latest_features_scaled)[0]
        
        return prediction
        
    def save_model(self, directory='models', filename=None):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No trained model to save")
            
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Create filename if not provided
        if filename is None:
            filename = f"sentiment_{self.model_type}_{self.algorithm}_model.pkl"
            
        # Save model and scaler
        model_path = os.path.join(directory, filename)
        scaler_path = os.path.join(directory, f"scaler_{filename}")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        return model_path
        
    def load_model(self, model_path, scaler_path):
        """Load a saved model"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        return self 