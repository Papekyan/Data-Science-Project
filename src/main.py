import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Import modules
from data_collection.news_collector import NewsCollector
from data_collection.social_media_collector import TwitterCollector
from data_collection.stock_data import StockDataCollector
from sentiment_analysis.sentiment_analyzer import SentimentAnalyzer
from prediction.sentiment_predictor import SentimentPredictor
from visualization.visualizer import SentimentVisualizer

# Load environment variables
load_dotenv()

class StockSentimentPredictor:
    """Main class for stock sentiment analysis and prediction"""
    
    def __init__(self, ticker, company_name=None, sentiment_method='vader', 
                prediction_type='classification', days_back=30):
        """
        Initialize the stock sentiment predictor
        
        Parameters:
        - ticker: Stock ticker symbol (e.g., "AAPL")
        - company_name: Company name (optional, will try to get from API if not provided)
        - sentiment_method: Method for sentiment analysis ('vader', 'textblob', 'finbert')
        - prediction_type: Type of prediction ('regression' or 'classification')
        - days_back: Number of days of historical data to collect
        """
        self.ticker = ticker
        self.days_back = days_back
        
        # Initialize collectors
        self.stock_collector = StockDataCollector()
        self.news_collector = NewsCollector()
        self.twitter_collector = None  # Initialize only if API keys are provided
        
        # Try to initialize Twitter collector if keys are available
        if all([os.getenv(key) for key in 
               ['TWITTER_API_KEY', 'TWITTER_API_SECRET', 
                'TWITTER_ACCESS_TOKEN', 'TWITTER_ACCESS_SECRET']]):
            try:
                self.twitter_collector = TwitterCollector()
            except Exception as e:
                print(f"Warning: Could not initialize Twitter collector: {e}")
        
        # Get company info if name not provided
        if company_name is None:
            company_info = self.stock_collector.get_company_info(ticker)
            self.company_name = company_info.get('name', ticker)
        else:
            self.company_name = company_name
            
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentAnalyzer(method=sentiment_method)
        
        # Initialize predictor
        self.predictor = SentimentPredictor(model_type=prediction_type)
        
        # Initialize visualizer
        self.visualizer = SentimentVisualizer()
        
        # Data containers
        self.stock_data = None
        self.news_data = None
        self.twitter_data = None
        self.sentiment_data = None
        self.merged_data = None
        
    def collect_data(self):
        """Collect all necessary data for analysis"""
        print(f"Collecting data for {self.ticker} ({self.company_name})...")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days_back)
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Collect stock data
        print("Collecting stock data...")
        self.stock_data = self.stock_collector.get_historical_data(
            self.ticker, start_date=start_date_str, end_date=end_date_str
        )
        
        # Collect news data
        print("Collecting news data...")
        self.news_data = self.news_collector.get_company_news(
            self.company_name, self.ticker, days_back=self.days_back
        )
        
        # Collect Twitter data if available
        if self.twitter_collector:
            print("Collecting Twitter data...")
            self.twitter_data = self.twitter_collector.get_stock_tweets(
                self.ticker, days_back=self.days_back
            )
            
        print("Data collection complete!")
        
    def analyze_sentiment(self):
        """Analyze sentiment of collected data"""
        if self.news_data is None or self.news_data.empty:
            raise ValueError("News data is empty. Run collect_data() first.")
            
        print("Analyzing sentiment...")
        
        # Analyze news sentiment
        news_sentiment = self.sentiment_analyzer.analyze_dataframe(
            self.news_data, 'content'
        )
        
        # Aggregate news sentiment by date
        agg_news_sentiment = self.sentiment_analyzer.aggregate_sentiment(
            news_sentiment, date_column='date', weight_column='popularity'
        )
        
        # Analyze Twitter sentiment if available
        if self.twitter_data is not None and not self.twitter_data.empty:
            twitter_sentiment = self.sentiment_analyzer.analyze_dataframe(
                self.twitter_data, 'text'
            )
            
            # Aggregate Twitter sentiment by date with followers as weight
            agg_twitter_sentiment = self.sentiment_analyzer.aggregate_sentiment(
                twitter_sentiment, date_column='date', weight_column='followers'
            )
            
            # Combine news and Twitter sentiment with equal weight
            self.sentiment_data = pd.merge(
                agg_news_sentiment, agg_twitter_sentiment, 
                on='date', how='outer', suffixes=('_news', '_twitter')
            )
            
            # Calculate combined sentiment
            self.sentiment_data['avg_sentiment'] = (
                self.sentiment_data['avg_sentiment_news'].fillna(0) * 0.7 + 
                self.sentiment_data['avg_sentiment_twitter'].fillna(0) * 0.3
            )
            
        else:
            # Use only news sentiment
            self.sentiment_data = agg_news_sentiment
            
        print("Sentiment analysis complete!")
        
    def train_model(self, test_size=0.2):
        """Train the prediction model"""
        if self.sentiment_data is None or self.sentiment_data.empty:
            raise ValueError("Sentiment data is empty. Run analyze_sentiment() first.")
            
        if self.stock_data is None or self.stock_data.empty:
            raise ValueError("Stock data is empty. Run collect_data() first.")
            
        print("Training prediction model...")
        
        # Train the model
        metrics = self.predictor.train(
            self.sentiment_data, self.stock_data, test_size=test_size
        )
        
        print("Model training complete!")
        print("Model performance metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
            
        return metrics
        
    def predict_next_day(self):
        """Predict the next day's price or movement"""
        if self.predictor.model is None:
            raise ValueError("Model is not trained. Run train_model() first.")
            
        print(f"Predicting next day for {self.ticker}...")
        
        # Make prediction
        prediction = self.predictor.predict_next_day(
            self.sentiment_data, self.stock_data
        )
        
        # Interpret prediction based on model type
        if self.predictor.model_type == 'regression':
            last_price = self.stock_data['Close'].iloc[-1]
            price_change = prediction - last_price
            percent_change = (price_change / last_price) * 100
            
            result = {
                'ticker': self.ticker,
                'last_price': last_price,
                'predicted_price': prediction,
                'price_change': price_change,
                'percent_change': percent_change
            }
            
            print(f"Prediction: {self.ticker} price will be ${prediction:.2f} (change: {price_change:.2f}, {percent_change:.2f}%)")
            
        else:  # classification
            movement = "UP" if prediction == 1 else "DOWN"
            probability = self.predictor.model.predict_proba(
                self.predictor.scaler.transform(self.predictor.latest_features_scaled)
            )[0][1]
            
            result = {
                'ticker': self.ticker,
                'predicted_movement': movement,
                'probability': probability
            }
            
            print(f"Prediction: {self.ticker} price will go {movement} with {probability:.2f} probability")
            
        return result
        
    def visualize_data(self, save_dir=None):
        """Generate visualizations"""
        if self.sentiment_data is None or self.stock_data is None:
            raise ValueError("Data is not available. Run collect_data() and analyze_sentiment() first.")
            
        print("Generating visualizations...")
        
        # Create directory for plots if provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        # Plot sentiment trend
        save_path = os.path.join(save_dir, f"{self.ticker}_sentiment_trend.png") if save_dir else None
        self.visualizer.plot_sentiment_trend(
            self.sentiment_data, ticker=self.ticker, save_path=save_path
        )
        
        # Plot sentiment vs. price
        save_path = os.path.join(save_dir, f"{self.ticker}_sentiment_vs_price.png") if save_dir else None
        self.visualizer.plot_sentiment_vs_price(
            self.sentiment_data, self.stock_data, ticker=self.ticker, save_path=save_path
        )
        
        # Plot sentiment distribution
        if 'sentiment_compound' in self.news_data.columns:
            save_path = os.path.join(save_dir, f"{self.ticker}_sentiment_distribution.png") if save_dir else None
            self.visualizer.plot_sentiment_distribution(
                self.news_data, sentiment_col='sentiment_compound', 
                ticker=self.ticker, save_path=save_path
            )
            
        print("Visualizations complete!")
        
    def save_results(self, output_dir='data'):
        """Save collected and processed data"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save stock data
        if self.stock_data is not None and not self.stock_data.empty:
            self.stock_data.to_csv(f"{output_dir}/{self.ticker}_stock_data.csv", index=False)
            
        # Save news data
        if self.news_data is not None and not self.news_data.empty:
            self.news_data.to_csv(f"{output_dir}/{self.ticker}_news_data.csv", index=False)
            
        # Save Twitter data
        if self.twitter_data is not None and not self.twitter_data.empty:
            self.twitter_data.to_csv(f"{output_dir}/{self.ticker}_twitter_data.csv", index=False)
            
        # Save sentiment data
        if self.sentiment_data is not None and not self.sentiment_data.empty:
            self.sentiment_data.to_csv(f"{output_dir}/{self.ticker}_sentiment_data.csv", index=False)
            
        # Save model
        if self.predictor.model is not None:
            self.predictor.save_model(
                directory='models', 
                filename=f"{self.ticker}_sentiment_predictor.pkl"
            )
            
        print(f"Results saved in '{output_dir}' directory")
        
    def run_full_analysis(self, save_results=True, visualize=True):
        """Run the complete analysis pipeline"""
        self.collect_data()
        self.analyze_sentiment()
        self.train_model()
        prediction = self.predict_next_day()
        
        if visualize:
            self.visualize_data(save_dir='data/plots')
            
        if save_results:
            self.save_results()
            
        return prediction

if __name__ == "__main__":
    # Example usage
    ticker = "AAPL"
    predictor = StockSentimentPredictor(ticker=ticker)
    predictor.run_full_analysis() 