import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from transformers import pipeline
from datetime import datetime
import torch
from collections import defaultdict


class DataProcessor:
    def __init__(self, market_data_files, news_data_file):
        """
        Initialize the DataProcessor with paths to data files

        Parameters:
        market_data_files (list): List of paths to market data CSV files
        news_data_file (str): Path to news data CSV file
        """
        self.market_data_files = market_data_files
        self.news_data_file = news_data_file
        self.market_scaler = MinMaxScaler()
        self.sentiment_analyzer = pipeline('sentiment-analysis')

    def load_market_data(self):
        """Load and combine market data from multiple files"""
        market_dfs = {}
        for file in self.market_data_files:
            symbol = file.split('_')[0]  # Extract symbol from filename
            df = pd.read_csv(file)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            market_dfs[symbol] = df

        # Combine all market data
        combined_market = pd.concat(market_dfs.values(), axis=1, keys=market_dfs.keys())
        return combined_market

    def process_market_features(self, market_data):
        """Process and engineer market features"""
        processed_data = market_data.copy()

        for symbol in market_data.columns.levels[0]:
            # Calculate technical indicators
            symbol_data = processed_data[symbol]

            # Adding Moving Averages
            symbol_data['MA5'] = symbol_data['Close'].rolling(window=5).mean()
            symbol_data['MA20'] = symbol_data['Close'].rolling(window=20).mean()

            # Adding price momentum
            symbol_data['Returns'] = symbol_data['Close'].pct_change()
            symbol_data['Volatility'] = symbol_data['Returns'].rolling(window=20).std()

            # Volume indicators
            symbol_data['Volume_MA5'] = symbol_data['Volume'].rolling(window=5).mean()

            # Price relative to moving averages
            symbol_data['Price_MA5_Ratio'] = symbol_data['Close'] / symbol_data['MA5']

        return processed_data.fillna(0)  # Fill NaN values with 0

    def process_news_data(self):
        """Process news data and calculate sentiment scores"""
        news_df = pd.read_csv(self.news_data_file)
        news_df['published_at'] = pd.to_datetime(news_df['published_at'])

        # Calculate sentiment scores for news titles and descriptions
        sentiments = defaultdict(list)
        for _, row in news_df.iterrows():
            date = row['published_at'].date()
            text = f"{row['title']} {row['description']}"
            sentiment = self.sentiment_analyzer(text)[0]
            sentiments[date].append(sentiment['score'] if sentiment['label'] == 'POSITIVE' else -sentiment['score'])

        # Average sentiment per day
        daily_sentiment = {date: np.mean(scores) for date, scores in sentiments.items()}
        sentiment_df = pd.DataFrame.from_dict(daily_sentiment, orient='index', columns=['sentiment'])
        sentiment_df.index = pd.to_datetime(sentiment_df.index)

        return sentiment_df

    def combine_and_normalize_data(self):
        """Combine and normalize all features for neural network training"""
        # Load and process all data
        market_data = self.load_market_data()
        processed_market = self.process_market_features(market_data)
        news_sentiment = self.process_news_data()

        # Combine market and news data
        combined_data = processed_market.merge(news_sentiment,
                                               left_index=True,
                                               right_index=True,
                                               how='left')

        # Fill missing sentiment values with 0
        combined_data['sentiment'] = combined_data['sentiment'].fillna(0)

        # Normalize numerical features
        numerical_columns = combined_data.select_dtypes(include=[np.number]).columns
        combined_data[numerical_columns] = self.market_scaler.fit_transform(combined_data[numerical_columns])

        return combined_data

    def prepare_training_sequences(self, data, sequence_length=10):
        """
        Prepare sequential data for training

        Parameters:
        data (DataFrame): Normalized combined data
        sequence_length (int): Length of input sequences

        Returns:
        X (tensor): Input sequences
        y (tensor): Target values
        """
        sequences = []
        targets = []

        for i in range(len(data) - sequence_length):
            sequence = data.iloc[i:i + sequence_length]
            target = data.iloc[i + sequence_length]['Returns']  # Predict next day's returns

            sequences.append(sequence.values)
            targets.append(target)

        # Convert to PyTorch tensors
        X = torch.FloatTensor(np.array(sequences))
        y = torch.FloatTensor(np.array(targets))

        return X, y

    def create_train_test_split(self, X, y, train_ratio=0.8):
        """Split data into training and testing sets"""
        split_idx = int(len(X) * train_ratio)

        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]

        return X_train, X_test, y_train, y_test


# Example usage:
def main():
    market_files = ['AAPL_price_history.csv', 'GOOGL_price_history.csv', 'MSFT_price_history.csv',
                    'AMZN_price_history.csv']
    news_file = 'news_data.csv'

    # Initialize data processor
    processor = DataProcessor(market_files, news_file)

    # Process and combine all data
    processed_data = processor.combine_and_normalize_data()

    # Prepare sequences for training
    X, y = processor.prepare_training_sequences(processed_data)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = processor.create_train_test_split(X, y)

    # Data is now ready for neural network training
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")


if __name__ == "__main__":
    main()