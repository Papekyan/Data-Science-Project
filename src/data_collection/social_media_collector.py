import os
import pandas as pd
import tweepy
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TwitterCollector:
    """Collects tweets related to stocks and companies"""
    
    def __init__(
        self, 
        api_key=None, 
        api_secret=None, 
        access_token=None, 
        access_secret=None
    ):
        """Initialize with Twitter API credentials"""
        # Use environment variables if not provided
        if api_key is None:
            api_key = os.getenv('TWITTER_API_KEY')
        if api_secret is None:
            api_secret = os.getenv('TWITTER_API_SECRET')
        if access_token is None:
            access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        if access_secret is None:
            access_secret = os.getenv('TWITTER_ACCESS_SECRET')
            
        # Validate credentials
        if not all([api_key, api_secret, access_token, access_secret]):
            raise ValueError("Twitter API credentials are required")
            
        # Authenticate
        auth = tweepy.OAuthHandler(api_key, api_secret)
        auth.set_access_token(access_token, access_secret)
        
        # Create API object
        self.api = tweepy.API(auth, wait_on_rate_limit=True)
        
    def get_stock_tweets(self, ticker, days_back=7, max_tweets=100):
        """
        Fetch tweets about a specific stock
        
        Parameters:
        - ticker: Stock ticker symbol (e.g., "AAPL")
        - days_back: How many days of tweets to collect
        - max_tweets: Maximum number of tweets to collect
        
        Returns:
        - DataFrame with tweets
        """
        # Calculate date for search
        since_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        # Create search query (include $ symbol and cashtag)
        query = f"${ticker} OR #{ticker} OR {ticker} stock"
        
        # Fetch tweets
        tweets = []
        try:
            # Use Cursor to handle pagination
            for tweet in tweepy.Cursor(
                self.api.search_tweets,
                q=query,
                lang="en",
                since=since_date,
                tweet_mode="extended"
            ).items(max_tweets):
                # Extract relevant information
                tweet_info = {
                    'id': tweet.id,
                    'created_at': tweet.created_at,
                    'text': tweet.full_text,
                    'user': tweet.user.screen_name,
                    'followers': tweet.user.followers_count,
                    'retweets': tweet.retweet_count,
                    'favorites': tweet.favorite_count
                }
                tweets.append(tweet_info)
                
            # Convert to DataFrame
            tweets_df = pd.DataFrame(tweets)
            
            # Add ticker information
            if not tweets_df.empty:
                tweets_df['ticker'] = ticker
                # Extract date from created_at
                tweets_df['date'] = pd.to_datetime(tweets_df['created_at']).dt.date
                
            return tweets_df
            
        except Exception as e:
            print(f"Error fetching tweets for {ticker}: {e}")
            return pd.DataFrame()
            
    def get_financial_influencer_tweets(self, influencers=None, days_back=7, max_tweets=100):
        """
        Fetch tweets from financial influencers
        
        Parameters:
        - influencers: List of Twitter handles of financial influencers
        - days_back: How many days of tweets to collect
        - max_tweets: Maximum number of tweets per influencer
        
        Returns:
        - DataFrame with tweets
        """
        if influencers is None:
            # Default list of financial influencers
            influencers = [
                'jimcramer', 'TheStalwart', 'ScottMinerd', 'ReformedBroker',
                'elerianm', 'carl_c_icahn', 'michael_saylor', 'wallstreetbets'
            ]
            
        all_tweets = []
        
        for influencer in influencers:
            try:
                # Get user timeline
                timeline = self.api.user_timeline(
                    screen_name=influencer,
                    count=max_tweets,
                    tweet_mode="extended"
                )
                
                # Filter by date
                since_date = datetime.now() - timedelta(days=days_back)
                
                for tweet in timeline:
                    if tweet.created_at >= since_date:
                        tweet_info = {
                            'id': tweet.id,
                            'created_at': tweet.created_at,
                            'text': tweet.full_text,
                            'user': tweet.user.screen_name,
                            'followers': tweet.user.followers_count,
                            'retweets': tweet.retweet_count,
                            'favorites': tweet.favorite_count
                        }
                        all_tweets.append(tweet_info)
                        
            except Exception as e:
                print(f"Error fetching tweets for {influencer}: {e}")
                continue
                
        # Convert to DataFrame
        tweets_df = pd.DataFrame(all_tweets)
        
        if not tweets_df.empty:
            # Extract date from created_at
            tweets_df['date'] = pd.to_datetime(tweets_df['created_at']).dt.date
            
        return tweets_df 