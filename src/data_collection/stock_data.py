import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

class StockDataCollector:
    """Collects historical and real-time stock price data"""
    
    def get_historical_data(self, ticker, start_date=None, end_date=None, period='1y'):
        """
        Fetch historical stock data
        
        Parameters:
        - ticker: Stock ticker symbol (e.g., "AAPL")
        - start_date: Start date for data (format: 'YYYY-MM-DD')
        - end_date: End date for data (format: 'YYYY-MM-DD')
        - period: Time period to download (default: '1y')
          Options: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
          (Only used if start_date and end_date are None)
        
        Returns:
        - DataFrame with historical stock data
        """
        try:
            # Create ticker object
            stock = yf.Ticker(ticker)
            
            # Fetch historical data
            if start_date and end_date:
                hist_data = stock.history(start=start_date, end=end_date)
            else:
                hist_data = stock.history(period=period)
                
            # Add ticker column
            hist_data['ticker'] = ticker
            
            # Reset index to make Date a column
            hist_data = hist_data.reset_index()
            
            # Format data
            if 'Date' in hist_data.columns:
                hist_data['date'] = pd.to_datetime(hist_data['Date']).dt.date
                
            return hist_data
            
        except Exception as e:
            print(f"Error fetching historical data for {ticker}: {e}")
            return pd.DataFrame()
            
    def get_multiple_stocks(self, tickers, period='1y'):
        """
        Fetch historical data for multiple stocks
        
        Parameters:
        - tickers: List of stock ticker symbols
        - period: Time period to download
        
        Returns:
        - Dictionary of DataFrames with historical stock data
        """
        stock_data = {}
        
        for ticker in tickers:
            data = self.get_historical_data(ticker, period=period)
            if not data.empty:
                stock_data[ticker] = data
                
        return stock_data
        
    def calculate_returns(self, data):
        """
        Calculate daily and cumulative returns
        
        Parameters:
        - data: DataFrame with stock price data
        
        Returns:
        - DataFrame with added return columns
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Calculate daily returns
        if 'Close' in df.columns:
            df['daily_return'] = df['Close'].pct_change()
            
            # Calculate cumulative returns
            df['cum_return'] = (1 + df['daily_return']).cumprod() - 1
            
        return df
        
    def get_company_info(self, ticker):
        """
        Get company information
        
        Parameters:
        - ticker: Stock ticker symbol
        
        Returns:
        - Dictionary with company information
        """
        try:
            # Create ticker object
            stock = yf.Ticker(ticker)
            
            # Get company info
            info = stock.info
            
            # Extract key information
            company_info = {
                'name': info.get('shortName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', None),
                'pe_ratio': info.get('trailingPE', None),
                'dividend_yield': info.get('dividendYield', None),
                'beta': info.get('beta', None),
                'description': info.get('longBusinessSummary', '')
            }
            
            return company_info
            
        except Exception as e:
            print(f"Error fetching company information for {ticker}: {e}")
            return {} 