import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class SentimentAnalyzer:
    """Analyzes sentiment of financial texts using various methods"""
    
    def __init__(self, method='vader'):
        """
        Initialize with specified sentiment analysis method
        
        Parameters:
        - method: Sentiment analysis method to use
          Options: 'vader', 'textblob', 'finbert'
        """
        self.method = method
        
        # Initialize appropriate analyzer based on method
        if method == 'vader':
            self.analyzer = SentimentIntensityAnalyzer()
        elif method == 'finbert':
            model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.analyzer = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        # For TextBlob, no initialization needed
        
    def analyze_text(self, text):
        """
        Analyze sentiment of a single text
        
        Parameters:
        - text: Text to analyze
        
        Returns:
        - Dictionary with sentiment scores
        """
        if not text or pd.isna(text):
            return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0}
            
        if self.method == 'vader':
            # VADER analysis
            scores = self.analyzer.polarity_scores(text)
            return scores
            
        elif self.method == 'textblob':
            # TextBlob analysis
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Convert to format similar to VADER for consistency
            return {
                'compound': polarity,
                'pos': max(0, polarity),
                'neg': max(0, -polarity),
                'neu': 1 - abs(polarity),
                'subjectivity': subjectivity
            }
            
        elif self.method == 'finbert':
            # FinBERT analysis
            try:
                # Handle text length limits
                if len(text) > 512:
                    text = text[:512]
                    
                result = self.analyzer(text)[0]
                label = result['label']
                score = result['score']
                
                # Convert to format similar to VADER
                compound = 0
                pos = 0
                neg = 0
                neu = 0
                
                if label == 'positive':
                    compound = score
                    pos = score
                elif label == 'negative':
                    compound = -score
                    neg = score
                elif label == 'neutral':
                    neu = score
                    
                return {
                    'compound': compound,
                    'pos': pos,
                    'neg': neg,
                    'neu': neu,
                    'finbert_label': label,
                    'finbert_score': score
                }
                
            except Exception as e:
                print(f"Error in FinBERT analysis: {e}")
                return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0}
                
    def analyze_dataframe(self, df, text_column):
        """
        Analyze sentiment for texts in a DataFrame
        
        Parameters:
        - df: DataFrame containing texts
        - text_column: Column name containing text to analyze
        
        Returns:
        - DataFrame with added sentiment columns
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Apply sentiment analysis to each text
        sentiment_data = []
        
        for text in result_df[text_column]:
            sentiment_data.append(self.analyze_text(text))
            
        # Convert results to DataFrame
        sentiment_df = pd.DataFrame(sentiment_data)
        
        # Concatenate with original DataFrame
        for col in sentiment_df.columns:
            result_df[f'sentiment_{col}'] = sentiment_df[col]
            
        return result_df
        
    def aggregate_sentiment(self, df, date_column='date', ticker_column=None, 
                            weight_column=None, sentiment_column='sentiment_compound'):
        """
        Aggregate sentiment scores by date and optionally by ticker
        
        Parameters:
        - df: DataFrame with sentiment scores
        - date_column: Column name for date
        - ticker_column: Column name for ticker (optional)
        - weight_column: Column name for weights (optional)
        - sentiment_column: Column name for sentiment scores
        
        Returns:
        - DataFrame with aggregated sentiment
        """
        # Group by columns
        group_cols = [date_column]
        if ticker_column and ticker_column in df.columns:
            group_cols.append(ticker_column)
            
        # Apply weights if specified
        if weight_column and weight_column in df.columns:
            # Weighted average
            def weighted_avg(group):
                weights = group[weight_column]
                values = group[sentiment_column]
                return np.average(values, weights=weights)
                
            agg_sentiment = df.groupby(group_cols).apply(weighted_avg).reset_index()
            agg_sentiment.columns = group_cols + ['avg_sentiment']
            
        else:
            # Simple average
            agg_sentiment = df.groupby(group_cols)[sentiment_column].mean().reset_index()
            agg_sentiment.columns = group_cols + ['avg_sentiment']
            
        return agg_sentiment 