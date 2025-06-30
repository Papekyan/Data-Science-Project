# Multi-Stock LSTM Prediction Model with Sentiment Analysis

A comprehensive data science project that combines Long Short-Term Memory (LSTM) neural networks with sentiment analysis to predict stock prices for major technology companies.

## 📊 Project Overview

This project implements an advanced machine learning approach to stock price prediction by integrating:
- **Technical Analysis**: Traditional stock indicators (RSI, MACD, Bollinger Bands)
- **Sentiment Analysis**: Market sentiment derived from financial news articles
- **Deep Learning**: LSTM neural networks for time series forecasting
- **Multi-Stock Analysis**: Simultaneous prediction of 4 major tech stocks

## 🎯 Stocks Analyzed

- **TSLA** (Tesla Inc.)
- **NVDA** (NVIDIA Corporation)  
- **AAPL** (Apple Inc.)
- **MSFT** (Microsoft Corporation)

## 📈 Methodology

### 1. Data Collection & Preprocessing
- **Stock Data**: Historical price data from Yahoo Finance (2010-2024)
- **Sentiment Data**: Financial news sentiment scores from FNSPID dataset (https://huggingface.co/datasets/Zihan1004/FNSPID)
- **Time Range**: January 1, 2010 to January 9, 2024 (aligned with Tesla's IPO)

### 2. Feature Engineering
Our model uses 12 features per prediction:
- **Technical Indicators (7 features)**:
  - Close Price
  - Volume
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Signal Line
  - Upper & Lower Bollinger Bands
- **Sentiment Features (1 feature)**:
  - Daily average sentiment score from financial news
- **Stock Identification (4 features)**:
  - One-hot encoding for each stock ticker

### 3. Model Architecture
- **Model Type**: Sequential LSTM Neural Network
- **Architecture**: 
  - LSTM Layer 1: 50 units with return sequences
  - LSTM Layer 2: 50 units with dropout (0.2)
  - LSTM Layer 3: 50 units with dropout (0.2)
  - Dense Output Layer: 1 unit (price prediction)
- **Sequence Length**: 60 days of historical data
- **Training Configuration**:
  - Optimizer: Adam
  - Loss Function: Mean Squared Error
  - Epochs: 30
  - Batch Size: 32

### 4. Data Splitting Strategy
- **Training Set**: 80% of historical data (temporal split)
- **Test Set**: 20% of most recent data
- **Validation**: Real-time validation during training

## 🔍 Key Results

### Model Performance Metrics

| Stock | MAE (USD) | RMSE (USD) | R² Score | Avg. Sentiment |
|-------|-----------|------------|----------|----------------|
| TSLA  | 9.66      | 12.77      | 0.9513   | 0.1229 ± 0.0281 |
| AAPL  | 5.42      | 7.09       | 0.8629   | 0.1231 ± 0.0279 |
| NVDA  | 6.00      | 9.52       | 0.2891   | 0.1231 ± 0.0279 |
| MSFT  | 14.70     | -          | -        | 0.1231 ± 0.0279 |

### Key Findings
- **Best Performance**: Tesla (TSLA) with R² = 0.9513, indicating excellent predictive accuracy
- **Consistent Accuracy**: Apple (AAPL) with low MAE of $5.42
- **Sentiment Impact**: Average positive sentiment (0.12+) across all stocks during the analysis period
- **Model Convergence**: Training loss decreased consistently over 30 epochs

## 📁 Project Structure

```
Data-Science-Project/
├── final_model.ipynb           # Main Jupyter notebook with complete analysis
├── plot.py                     # Plotting utilities (currently empty)
├── sentiment_by_day.csv        # Processed daily sentiment scores
├── README.md                   # This documentation
├── plots/                      # Generated visualizations
│   ├── final_prediction_AAPL.png
│   ├── final_prediction_MSFT.png
│   ├── final_prediction_NVDA.png
│   ├── final_prediction_TSLA.png
│   ├── training_loss_detail_with_sentiment.png
│   └── training_validation_loss_with_sentiment.png
└── venv/                       # Python virtual environment
```

## 🛠️ Technical Implementation

### Data Processing Pipeline
1. **Stock Data Loading**: Automated download from Yahoo Finance API
2. **Technical Indicator Calculation**: RSI, MACD, Bollinger Bands computation
3. **Sentiment Integration**: Date-based joining of news sentiment with stock data
4. **Feature Scaling**: MinMaxScaler normalization for neural network compatibility
5. **Sequence Generation**: Creation of 60-day sliding windows for LSTM input

### Sentiment Analysis Integration
- **Source**: FNSPID (Financial News Sentiment and Price Impact Dataset)
- **Method**: VADER sentiment analysis on financial news headlines
- **Aggregation**: Daily average sentiment scores
- **Integration**: Inner join with stock data on trading dates

## 📊 Visualizations

The project generates several key visualizations:
- **Stock Price Predictions**: Actual vs. predicted prices for each stock
- **Training Progress**: Loss curves showing model convergence
- **Performance Metrics**: Comprehensive evaluation charts

## 🚀 How to Run

1. **Environment Setup**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run the Analysis**:
   ```bash
   jupyter notebook final_model.ipynb
   ```

3. **Execute All Cells**: Run the notebook from top to bottom to reproduce results

## 🔧 Dependencies

- `yfinance`: Stock data retrieval
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `scikit-learn`: Data preprocessing and metrics
- `tensorflow/keras`: Deep learning framework
- `matplotlib`: Visualization
- `nltk`: Natural language processing for sentiment analysis

## 💡 Key Innovations

1. **Multi-Stock Architecture**: Single model predicting multiple stocks simultaneously
2. **Sentiment Integration**: Novel incorporation of daily news sentiment
3. **Comprehensive Feature Set**: Combination of technical and sentiment indicators
4. **Temporal Data Splitting**: Proper time-series validation methodology

## 📋 Future Improvements

- **Real-time Prediction**: Live data streaming integration
- **Additional Features**: Economic indicators, social media sentiment
- **Model Optimization**: Hyperparameter tuning and architecture search
- **Risk Analysis**: Volatility prediction and confidence intervals

## 👥 Authors

This project was developed as part of a comprehensive data science analysis combining traditional financial modeling with modern machine learning techniques.

---

Financial indicators - Konstantinos Papanagiotou
Sentiment Analysis - Jannick Drechsler
Building the modell - Simon Papekyan 

Data Selection - Jannick Drechsler, Konstantinos Papanagiotou, Simon Papekyan
