#!/usr/bin/env python
"""
Stock Sentiment Analysis Example

This script demonstrates how to use the Stock Sentiment Predictor to analyze
sentiment from news and predict stock movements.
"""

import os
from src.main import StockSentimentPredictor
from dotenv import load_dotenv





# Load environment variables
load_dotenv()

def main():
    """Run a stock sentiment analysis example"""
    
    # Define the stock to analyze
    ticker = "AAPL"
    
    # Create the predictor (with default parameters)
    predictor = StockSentimentPredictor(
        ticker=ticker,
        sentiment_method='vader',
        prediction_type='classification',
        days_back=30
    )
    
    print(f"Running sentiment analysis for {ticker}...")
    
    # Collect data
    predictor.collect_data()
    
    # Analyze sentiment
    predictor.analyze_sentiment()
    
    # Visualize data (creates plots)
    predictor.visualize_data(save_dir='data/plots')
    
    # Train prediction model
    metrics = predictor.train_model()
    
    # Make prediction for next day
    prediction = predictor.predict_next_day()
    
    # Save results
    predictor.save_results()
    
    print("\nAnalysis complete!")
    print(f"Results saved in the 'data' directory")
    print(f"Plots saved in the 'data/plots' directory")
    
    # Return prediction result
    return prediction

if __name__ == "__main__":
    prediction = main()
    
    # Print the prediction
    print("\nPrediction Result:")
    for key, value in prediction.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}") 