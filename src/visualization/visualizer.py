import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.ticker as mticker

class SentimentVisualizer:
    """Visualizes sentiment analysis and stock prediction results"""
    
    def __init__(self, style='darkgrid', figsize=(12, 8)):
        """
        Initialize with plot style and size settings
        
        Parameters:
        - style: Seaborn style ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks')
        - figsize: Default figure size as tuple (width, height)
        """
        self.figsize = figsize
        sns.set_style(style)
        
    def plot_sentiment_trend(self, sentiment_df, date_col='date', sentiment_col='avg_sentiment', 
                             ticker=None, title=None, save_path=None):
        """
        Plot sentiment trend over time
        
        Parameters:
        - sentiment_df: DataFrame with sentiment data
        - date_col: Column name for date
        - sentiment_col: Column name for sentiment
        - ticker: Stock ticker (optional, for title)
        - title: Custom title (optional)
        - save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=self.figsize)
        
        # Convert date to datetime if it's not
        if not pd.api.types.is_datetime64_dtype(sentiment_df[date_col]):
            sentiment_df = sentiment_df.copy()
            sentiment_df[date_col] = pd.to_datetime(sentiment_df[date_col])
            
        # Plot sentiment trend
        ax = plt.plot(sentiment_df[date_col], sentiment_df[sentiment_col], 
                      marker='o', linestyle='-', color='#1f77b4')
        
        # Add a horizontal line at y=0
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Format dates on x-axis
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # Set title and labels
        if title is None:
            if ticker:
                title = f"Sentiment Trend for {ticker}"
            else:
                title = "Sentiment Trend Over Time"
                
        plt.title(title, fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Sentiment Score', fontsize=12)
        
        # Add grid and tight layout
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def plot_sentiment_vs_price(self, sentiment_df, stock_df, date_col='date', 
                               sentiment_col='avg_sentiment', price_col='Close', 
                               ticker=None, title=None, save_path=None):
        """
        Plot sentiment and stock price on the same timeline
        
        Parameters:
        - sentiment_df: DataFrame with sentiment data
        - stock_df: DataFrame with stock price data
        - date_col: Column name for date
        - sentiment_col: Column name for sentiment
        - price_col: Column name for stock price
        - ticker: Stock ticker (optional, for title)
        - title: Custom title (optional)
        - save_path: Path to save the plot (optional)
        """
        # Merge data
        merged_df = pd.merge(sentiment_df, stock_df, on=date_col, how='inner')
        
        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=self.figsize)
        
        # Plot sentiment
        ax1.plot(merged_df[date_col], merged_df[sentiment_col], 
                marker='o', linestyle='-', color='blue', label='Sentiment')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Sentiment Score', color='blue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Add a horizontal line at y=0 for sentiment
        ax1.axhline(y=0, color='blue', linestyle='--', alpha=0.3)
        
        # Create second y-axis for stock price
        ax2 = ax1.twinx()
        ax2.plot(merged_df[date_col], merged_df[price_col], 
                marker='s', linestyle='-', color='green', label='Stock Price')
        ax2.set_ylabel('Stock Price', color='green', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='green')
        
        # Format dates on x-axis
        ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # Set title
        if title is None:
            if ticker:
                title = f"Sentiment vs. Price for {ticker}"
            else:
                title = "Sentiment vs. Stock Price"
                
        plt.title(title, fontsize=16)
        
        # Add legends for both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        # Add grid and tight layout
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def plot_correlation_heatmap(self, data, title="Feature Correlation Heatmap", 
                                save_path=None):
        """
        Plot correlation heatmap of features
        
        Parameters:
        - data: DataFrame with features
        - title: Plot title
        - save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=self.figsize)
        
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                   linewidths=0.5, vmin=-1, vmax=1)
        
        # Set title
        plt.title(title, fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def plot_prediction_vs_actual(self, dates, actual, predicted, ticker=None, 
                                 title=None, save_path=None):
        """
        Plot predicted vs. actual stock prices
        
        Parameters:
        - dates: Array of dates
        - actual: Array of actual prices/values
        - predicted: Array of predicted prices/values
        - ticker: Stock ticker (optional, for title)
        - title: Custom title (optional)
        - save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=self.figsize)
        
        # Plot actual and predicted values
        plt.plot(dates, actual, marker='o', linestyle='-', color='blue', label='Actual')
        plt.plot(dates, predicted, marker='x', linestyle='--', color='red', label='Predicted')
        
        # Format dates on x-axis
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # Set title and labels
        if title is None:
            if ticker:
                title = f"Predicted vs. Actual Values for {ticker}"
            else:
                title = "Predicted vs. Actual Values"
                
        plt.title(title, fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        
        # Add legend
        plt.legend()
        
        # Add grid and tight layout
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def plot_sentiment_distribution(self, sentiment_df, sentiment_col='sentiment_compound', 
                                   ticker=None, title=None, save_path=None):
        """
        Plot distribution of sentiment values
        
        Parameters:
        - sentiment_df: DataFrame with sentiment data
        - sentiment_col: Column name for sentiment
        - ticker: Stock ticker (optional, for title)
        - title: Custom title (optional)
        - save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=self.figsize)
        
        # Plot distribution
        sns.histplot(sentiment_df[sentiment_col], kde=True, bins=30)
        
        # Add vertical line at mean
        mean_val = sentiment_df[sentiment_col].mean()
        plt.axvline(x=mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        
        # Add vertical line at 0
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Set title and labels
        if title is None:
            if ticker:
                title = f"Sentiment Distribution for {ticker}"
            else:
                title = "Sentiment Distribution"
                
        plt.title(title, fontsize=16)
        plt.xlabel('Sentiment Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        
        # Add legend
        plt.legend()
        
        # Add grid and tight layout
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show() 