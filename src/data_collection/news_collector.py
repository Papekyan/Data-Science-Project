import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from newsapi import NewsApiClient

# Load environment variables
load_dotenv()

class NewsCollector:
    """Collects news articles related to a specific stock or company"""
    
    def __init__(self, api_key=None):
        """Initialize with News API key"""
        if api_key is None:
            api_key = os.getenv('NEWS_API_KEY')
            
        if not api_key:
            raise ValueError("News API key is required")
            
        self.api = NewsApiClient(api_key=api_key)
        
    def get_company_news(self, company_name, ticker, days_back=7):
        """
        Fetch news for a specific company
        
        Parameters:
        - company_name: Name of the company (e.g., "Apple")
        - ticker: Stock ticker symbol (e.g., "AAPL")
        - days_back: How many days of news to collect
        
        Returns:
        - DataFrame with news articles
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Format dates for API
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')
        
        # Create search query (company name OR ticker)
        query = f'"{company_name}" OR "{ticker}"'
        
        # Fetch news
        news = self.api.get_everything(
            q=query,
            from_param=from_date,
            to=to_date,
            language='en',
            sort_by='publishedAt'
        )
        
        # Convert to DataFrame
        if news['status'] == 'ok' and news['totalResults'] > 0:
            articles_df = pd.DataFrame(news['articles'])
            
            # Extract date from publishedAt
            articles_df['date'] = pd.to_datetime(articles_df['publishedAt']).dt.date
            
            # Add company and ticker information
            articles_df['company'] = company_name
            articles_df['ticker'] = ticker
            
            return articles_df
        else:
            print(f"No news found for {company_name} ({ticker})")
            return pd.DataFrame()
            
    def get_sector_news(self, sector, days_back=7):
        """Get news for an entire sector"""
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Format dates for API
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')
        
        # Fetch sector news
        news = self.api.get_everything(
            q=f'"{sector}" AND (finance OR stock OR market OR investment)',
            from_param=from_date,
            to=to_date,
            language='en',
            sort_by='publishedAt'
        )
        
        # Convert to DataFrame
        if news['status'] == 'ok' and news['totalResults'] > 0:
            articles_df = pd.DataFrame(news['articles'])
            
            # Extract date from publishedAt
            articles_df['date'] = pd.to_datetime(articles_df['publishedAt']).dt.date
            
            # Add sector information
            articles_df['sector'] = sector
            
            return articles_df
        else:
            print(f"No news found for sector: {sector}")
            return pd.DataFrame() 