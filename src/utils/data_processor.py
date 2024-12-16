import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from transformers import pipeline
import logging
import os
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

        logging.info(f"Attempting to load market data files: {self.market_data_files}")

        for file in self.market_data_files:
            try:
                # Extract symbol from filename
                symbol = os.path.basename(file).split('_')[0]
                logging.info(f"Processing file for symbol {symbol}: {file}")

                # Read the CSV file
                df = pd.read_csv(file)
                logging.info(f"Successfully read file {file}")

                # Convert to datetime
                if 'Date' not in df.columns:
                    raise ValueError(f"No 'Date' column found in {file}")

                # Handle datetime conversion
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.dropna(subset=['Date'])

                if df.empty:
                    raise ValueError(f"No valid data after date processing in {file}")

                # Set index
                df.set_index('Date', inplace=True)

                # Convert numeric columns to float64
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].astype('float64')

                # Store processed dataframe
                if symbol in market_dfs:
                    # If we already have data for this symbol, concatenate
                    market_dfs[symbol] = pd.concat([market_dfs[symbol], df])
                else:
                    market_dfs[symbol] = df

                logging.info(f"Successfully processed data for {symbol}")

            except Exception as e:
                logging.error(f"Error processing file {file}: {str(e)}")
                continue

        if not market_dfs:
            raise ValueError("No market data could be processed")

        # Combine all market data
        combined_market = pd.concat(market_dfs.values(), axis=1, keys=market_dfs.keys())
        logging.info(f"Successfully combined market data from {len(market_dfs)} symbols")
        logging.info(f"Data range: {combined_market.index.min()} to {combined_market.index.max()}")

        return combined_market

    def calculate_returns(self, market_data):
        """Calculate returns for each stock"""
        returns_data = pd.DataFrame(index=market_data.index)

        # Handle multi-index columns if present
        if isinstance(market_data.columns, pd.MultiIndex):
            # Get unique symbols from the first level of the multi-index
            symbols = market_data.columns.get_level_values(0).unique()
        else:
            # Fall back to original method for single-index columns
            symbols = list(set([str(col).split('_')[0] for col in market_data.columns]))

        for symbol in symbols:
            try:
                # Handle multi-index columns
                if isinstance(market_data.columns, pd.MultiIndex):
                    close_prices = market_data[symbol]['Close']
                else:
                    close_prices = market_data[f"{symbol}_Close"]

                # Calculate returns
                returns_data[f"{symbol}_Returns"] = close_prices.pct_change()
                returns_data[f"{symbol}_Returns_5d"] = close_prices.pct_change(periods=5)
                returns_data[f"{symbol}_Returns_20d"] = close_prices.pct_change(periods=20)
                returns_data[f"{symbol}_Volatility"] = close_prices.pct_change().rolling(window=20).std()

                logging.info(f"Calculated returns for {symbol}")

            except Exception as e:
                logging.error(f"Error calculating returns for {symbol}: {str(e)}")
                continue

        # Drop rows with NaN values that occur at the beginning due to the calculations
        returns_data = returns_data.dropna()

        if returns_data.empty:
            raise ValueError("No valid returns data could be calculated")

        logging.info(f"Calculated returns for {len(symbols)} symbols")
        logging.info(f"Returns columns: {returns_data.columns.tolist()}")

        return returns_data

    def process_news_data(self):
        """Process news data and calculate sentiment scores"""
        try:
            # Read news data
            news_df = pd.read_csv(self.news_data_file)
            print("News data types:", news_df.dtypes)
            print("First few rows of news data:", news_df.head())

            # Convert publication date to datetime
            news_df['webPublicationDate'] = pd.to_datetime(news_df['webPublicationDate'])
            news_df.set_index('webPublicationDate', inplace=True)

            # Calculate daily sentiment scores
            daily_sentiment = pd.DataFrame(index=pd.date_range(
                start=news_df.index.min(),
                end=news_df.index.max(),
                freq='D'
            ))

            # Process sentiment for each day's articles
            for date in daily_sentiment.index:
                date_articles = news_df[news_df.index.date == date.date()]
                if not date_articles.empty:
                    titles = date_articles['webTitle'].tolist()
                    sentiments = self.sentiment_analyzer(titles)
                    scores = [1 if s['label'] == 'POSITIVE' else 0 for s in sentiments]

                    daily_sentiment.loc[date, 'sentiment_mean'] = np.mean(scores)
                    daily_sentiment.loc[date, 'sentiment_std'] = np.std(scores)
                    daily_sentiment.loc[date, 'news_volume'] = len(titles)

            # Fill missing values and calculate additional features
            daily_sentiment = daily_sentiment.fillna(method='ffill').fillna(0)
            daily_sentiment['sentiment_volatility'] = daily_sentiment['sentiment_std'].rolling(window=5).std()
            daily_sentiment['political_impact_score'] = daily_sentiment['sentiment_mean'] * daily_sentiment[
                'news_volume']
            daily_sentiment['political_impact_ma60'] = daily_sentiment['political_impact_score'].rolling(
                window=60).mean()

            return daily_sentiment

        except Exception as e:
            logging.error(f"Error processing news data: {str(e)}")
            raise

    def combine_and_normalize_data(self):
        """Load, combine, and normalize market and news data"""
        try:
            # Load market data
            market_data = self.load_market_data()
            logging.info("Market data columns structure:")
            logging.info(market_data.columns)

            # Calculate returns
            returns_data = self.calculate_returns(market_data)

            # Load and process news data
            news_features = self.process_news_data()

            # Reset market_data index if it's timezone aware
            if market_data.index.tz is not None:
                market_data.index = market_data.index.tz_localize(None)

            # Reset returns_data index if it's timezone aware
            if returns_data.index.tz is not None:
                returns_data.index = returns_data.index.tz_localize(None)

            # Reset news_features index if it's timezone aware
            if news_features.index.tz is not None:
                news_features.index = news_features.index.tz_localize(None)

            # Combine all features
            combined_data = pd.concat([market_data, returns_data, news_features], axis=1)

            # Fill any missing values
            combined_data = combined_data.fillna(method='ffill').fillna(method='bfill')

            print("Final combined features:", combined_data.columns.tolist())

            return combined_data

        except Exception as e:
            logging.error(f"Error in combine_and_normalize_data: {str(e)}")
            raise

    def prepare_training_sequences(self, data, sequence_length=60):
        """Prepare sequences for training"""
        # Identify returns columns
        returns_columns = [col for col in data.columns if 'Returns' in col]
        if not returns_columns:
            raise ValueError("No Returns columns found in the data")

        # Prepare feature columns (everything except returns)
        feature_columns = [col for col in data.columns if col not in returns_columns]

        # Create sequences
        X = []
        y = []

        # Normalize features
        scaler = MinMaxScaler()
        features_normalized = scaler.fit_transform(data[feature_columns])
        features_df = pd.DataFrame(features_normalized, columns=feature_columns, index=data.index)

        # Combine normalized features with returns
        full_data = pd.concat([features_df, data[returns_columns]], axis=1)

        # Create sequences
        for i in range(len(full_data) - sequence_length):
            # Feature sequence
            X.append(full_data[feature_columns].iloc[i:(i + sequence_length)].values)

            # Target (next day returns)
            y.append(full_data[returns_columns].iloc[i + sequence_length].values)

        return np.array(X), np.array(y), returns_columns

    def create_train_test_split(self, X, y, train_size=0.7, val_size=0.15):
        """Create train-validation-test split"""
        n = len(X)
        train_end = int(n * train_size)
        val_end = int(n * (train_size + val_size))

        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]

        y_train = y[:train_end]
        y_val = y[train_end:val_end]
        y_test = y[val_end:]

        return X_train, X_val, X_test, y_train, y_val, y_test