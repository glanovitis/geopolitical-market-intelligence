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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sentiment_analyzer = pipeline('sentiment-analysis',
                                           model='distilbert/distilbert-base-uncased-finetuned-sst-2-english',
                                           device=device)

    def load_market_data(self):
        """Load and combine market data from multiple files"""
        market_dfs = {}
        for file in self.market_data_files:
            symbol = file.split('_')[0]  # Extract symbol from filename
            df = pd.read_csv(file)
            # Convert to datetime and ensure timezone naive
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            df.set_index('Date', inplace=True)
            # Convert all numeric columns to float64
            df = df.astype('float64')
            market_dfs[symbol] = df

        # Combine all market data
        combined_market = pd.concat(market_dfs.values(), axis=1, keys=market_dfs.keys())
        return combined_market

    def process_market_features(self, market_data):
        """Process and engineer market features"""
        processed_data = market_data.copy()

        for symbol in market_data.columns.levels[0]:
            # Create a view of the data using .loc and ensure float64 dtype
            symbol_data = processed_data.loc[:, symbol].astype('float64').copy()

            # Calculate technical indicators
            symbol_data['MA5'] = symbol_data['Close'].rolling(window=5).mean()
            symbol_data['MA20'] = symbol_data['Close'].rolling(window=20).mean()
            symbol_data['Returns'] = symbol_data['Close'].pct_change()
            symbol_data['Volatility'] = symbol_data['Returns'].rolling(window=20).std()
            symbol_data['Volume_MA5'] = symbol_data['Volume'].rolling(window=5).mean()
            symbol_data['Price_MA5_Ratio'] = symbol_data['Close'] / symbol_data['MA5']

            # Update the original data with new columns
            for col in symbol_data.columns:
                processed_data.loc[:, (symbol, col)] = symbol_data[col]

        return processed_data.fillna(0)

    def process_news_data(self):
        """Process news data and calculate sentiment scores"""
        news_df = pd.read_csv(self.news_data_file)
        # Convert to datetime and make it timezone naive
        news_df['published_at'] = pd.to_datetime(news_df['published_at']).dt.tz_localize(None)

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

        # Flatten the multi-level columns before merging
        processed_market_flat = processed_market.copy()
        # Join column names with underscore, preserving all column names including technical indicators
        processed_market_flat.columns = ['_'.join(col).strip() for col in processed_market_flat.columns.values]

        # Debug print
        print("Available columns after flattening:", processed_market_flat.columns.tolist())

        # Combine market and news data
        combined_data = processed_market_flat.merge(news_sentiment,
                                                    left_index=True,
                                                    right_index=True,
                                                    how='left')

        # Fill missing sentiment values with 0
        combined_data['sentiment'] = combined_data['sentiment'].fillna(0)

        # Normalize numerical features
        numerical_columns = combined_data.select_dtypes(include=[np.number]).columns
        combined_data[numerical_columns] = self.market_scaler.fit_transform(combined_data[numerical_columns])

        # Debug print
        print("Final columns:", combined_data.columns.tolist())

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

        # Debug print to see available columns
        print("Available columns for training:", data.columns.tolist())

        # Find Returns columns for each symbol
        returns_columns = [col for col in data.columns if 'Returns' in col]
        if not returns_columns:
            raise ValueError("No Returns columns found in the data. Available columns: " + str(data.columns.tolist()))

        # Use the first symbol's Returns as target (you might want to modify this based on your needs)
        target_column = returns_columns[0]
        print(f"Using {target_column} as target variable")

        for i in range(len(data) - sequence_length):
            sequence = data.iloc[i:i + sequence_length]
            target = data.iloc[i + sequence_length][target_column]

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