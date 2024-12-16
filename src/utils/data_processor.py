import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
        self.base_dir = os.path.dirname(os.path.dirname(market_data_files[0]))
        self.scaler = None
        logging.info(f"Initialized DataProcessor with base directory: {self.base_dir}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sentiment_analyzer = pipeline('sentiment-analysis',
                                           model='distilbert/distilbert-base-uncased-finetuned-sst-2-english',
                                           device=device)

    def load_or_process_data(self):
        """Load processed data from cache or process it fresh"""
        processed_data_path = os.path.join(self.base_dir, 'processed_data.pkl')

        if os.path.exists(processed_data_path):
            processed_time = os.path.getmtime(processed_data_path)
            source_files_time = max(
                max(os.path.getmtime(f) for f in self.market_data_files),
                os.path.getmtime(self.news_data_file)
            )

            if processed_time > source_files_time:
                logging.info("Loading preprocessed data from cache...")
                try:
                    data = pd.read_pickle(processed_data_path)
                    # Handle NaN values
                    self._handle_missing_values(data)
                    return data
                except Exception as e:
                    logging.warning(f"Error loading cached data: {e}. Will reprocess.")

        # Process data if cache doesn't exist or is outdated
        return self.combine_and_normalize_data()

    def _handle_missing_values(self, data):
        """Handle missing values in the dataset"""
        # Check for NaN values
        nan_cols = data.columns[data.isna().any()].tolist()
        if nan_cols:
            logging.warning(f"NaN values found in columns: {nan_cols}")

            # Fill NaN values using forward fill first
            data.fillna(method='ffill', inplace=True)

            # Then use backward fill for any remaining NaNs at the beginning
            data.fillna(method='bfill', inplace=True)

            # If there are still any NaNs, fill them with 0
            if data.isna().any().any():
                logging.warning("Some NaN values remain after forward/backward filling. Filling with 0.")
                data.fillna(0, inplace=True)

            logging.info("Missing values handled")

        return data

    def load_or_create_sequences(self, processed_data, sequence_length):
        """Load sequences from cache or create them fresh"""
        sequences_path = os.path.join(self.base_dir, f'sequences_len{sequence_length}.npz')

        if os.path.exists(sequences_path):
            processed_time = os.path.getmtime(sequences_path)
            data_time = os.path.getmtime(os.path.join(self.base_dir, 'processed_data.pkl'))

            if processed_time > data_time:
                logging.info("Loading preprocessed sequences from cache...")
                try:
                    with np.load(sequences_path, allow_pickle=True) as data:
                        return data['X'], data['y'], data['returns_columns']
                except Exception as e:
                    logging.warning(f"Error loading cached sequences: {e}. Will reprocess.")

        # Create sequences if cache doesn't exist or is outdated
        logging.info("Creating sequences from processed data...")
        X, y, returns_columns = self.prepare_training_sequences(
            processed_data,
            sequence_length=sequence_length
        )

        # Save sequences
        try:
            np.savez_compressed(
                sequences_path,
                X=X,
                y=y,
                returns_columns=returns_columns
            )
            logging.info(f"Saved sequences to {sequences_path}")
        except Exception as e:
            logging.warning(f"Error saving sequences: {e}")

        return X, y, returns_columns

    def _normalize_date_index(self, df):
        """
        Normalize datetime index to date only (remove time component)
        """
        try:
            # Convert index to datetime if it's not already
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # Convert to date only (removes time component)
            df.index = df.index.date
            # Convert back to datetime for consistency, but without time component
            df.index = pd.to_datetime(df.index)

            return df
        except Exception as e:
            logging.error(f"Error normalizing date index: {e}")
            raise

    def _convert_datetime_index(self, df):
        """
        Convert index to datetime and handle timezone-aware datetimes properly

        Parameters:
        df (pd.DataFrame): DataFrame with datetime index

        Returns:
        pd.DataFrame: DataFrame with naive datetime index
        """
        try:
            if df.index.tz is not None:
                # Convert to UTC first, then convert to naive datetime
                df.index = df.index.tz_convert('UTC').tz_localize(None)
            return df
        except AttributeError:
            # If index is not already datetime type
            df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
            return df

    def load_market_data(self):
        """Load and combine market data from multiple files"""
        market_dfs = {}

        for file in self.market_data_files:
            try:
                symbol = os.path.basename(file).split('_')[0]
                df = pd.read_csv(file)

                # Debug the date column
                logging.info(f"Sample dates from {symbol}: {df['Date'].head().tolist()}")

                # Convert dates safely
                try:
                    df['Date'] = pd.to_datetime(df['Date'], utc=True)
                    # Remove time component
                    df['Date'] = df['Date'].dt.normalize()

                    # Filter out future dates
                    current_date = pd.Timestamp.now(tz='UTC').normalize()
                    df = df[df['Date'] <= current_date]

                    if df.empty:
                        logging.warning(f"No valid dates for {symbol} after filtering")
                        continue

                    df.set_index('Date', inplace=True)

                    # Store processed dataframe
                    if symbol in market_dfs:
                        market_dfs[symbol] = pd.concat([market_dfs[symbol], df])
                    else:
                        market_dfs[symbol] = df

                    logging.info(f"Processed {symbol} data: {df.index.min()} to {df.index.max()}")

                except Exception as e:
                    logging.error(f"Error processing dates for {symbol}: {str(e)}")
                    continue

            except Exception as e:
                logging.error(f"Error processing file {file}: {str(e)}")
                continue

        if not market_dfs:
            raise ValueError("No market data could be processed")

        # Combine all market data and sort by date
        combined_market = pd.concat(market_dfs.values(), axis=1, keys=market_dfs.keys())
        combined_market = combined_market.sort_index()

        # Log the structure of the combined data
        logging.info(f"Combined market data shape: {combined_market.shape}")
        logging.info(f"Combined market data columns: {combined_market.columns.tolist()}")
        logging.info(f"Date range: {combined_market.index.min()} to {combined_market.index.max()}")

        return combined_market

    def calculate_returns(self, market_data):
        """Calculate returns for each stock"""
        try:
            returns_data = pd.DataFrame(index=market_data.index)

            # Log the structure of input data
            logging.info("Calculating returns from market data:")
            logging.info(f"Market data shape: {market_data.shape}")
            logging.info(f"Market data columns: {market_data.columns.tolist()}")

            # Get unique symbols from multi-index columns
            if isinstance(market_data.columns, pd.MultiIndex):
                symbols = market_data.columns.get_level_values(0).unique()
            else:
                symbols = list(set([col.split('_')[0] for col in market_data.columns if 'Close' in col]))

            logging.info(f"Processing returns for symbols: {symbols}")

            for symbol in symbols:
                try:
                    # Handle multi-index columns
                    if isinstance(market_data.columns, pd.MultiIndex):
                        close_prices = market_data[symbol]['Close']
                    else:
                        close_prices = market_data[f"{symbol}_Close"]

                    # Calculate basic returns
                    returns_data[f"{symbol}_Returns"] = close_prices.pct_change()

                    # Calculate additional features only if we have enough data
                    if len(close_prices) >= 5:
                        returns_data[f"{symbol}_Returns_5d"] = close_prices.pct_change(periods=5)
                    if len(close_prices) >= 20:
                        returns_data[f"{symbol}_Returns_20d"] = close_prices.pct_change(periods=20)
                        returns_data[f"{symbol}_Volatility"] = (
                            close_prices.pct_change().rolling(window=20).std()
                        )

                    logging.info(f"Successfully calculated returns for {symbol}")

                except Exception as e:
                    logging.error(f"Error calculating returns for {symbol}: {str(e)}")
                    continue

            # Remove rows with all NaN values
            returns_data = returns_data.dropna(how='all')

            if returns_data.empty:
                raise ValueError("No valid returns data could be calculated")

            # Log the final returns data structure
            logging.info(f"Returns data shape: {returns_data.shape}")
            logging.info(f"Returns columns: {returns_data.columns.tolist()}")
            logging.info(f"Returns date range: {returns_data.index.min()} to {returns_data.index.max()}")

            return returns_data

        except Exception as e:
            logging.error(f"Error in calculate_returns: {str(e)}")
            raise

    def process_news_data(self):
        """Process news data and calculate sentiment scores"""
        try:
            news_df = pd.read_csv(self.news_data_file)

            # Convert to datetime and normalize to date only
            news_df['webPublicationDate'] = pd.to_datetime(news_df['webPublicationDate']).dt.date
            news_df['webPublicationDate'] = pd.to_datetime(news_df['webPublicationDate'])

            news_df.set_index('webPublicationDate', inplace=True)

            # Create date range with only dates (no time component)
            date_range = pd.date_range(
                start=news_df.index.min().date(),
                end=news_df.index.max().date(),
                freq='D'
            )

            daily_sentiment = pd.DataFrame(index=date_range)

            # Process sentiments by date
            for date in date_range:
                date_articles = news_df[news_df.index.date == date.date()]
                if not date_articles.empty:
                    titles = date_articles['webTitle'].tolist()
                    sentiments = self.sentiment_analyzer(titles)
                    scores = [1 if s['label'] == 'POSITIVE' else 0 for s in sentiments]

                    daily_sentiment.loc[date, 'sentiment_mean'] = np.mean(scores)
                    daily_sentiment.loc[date, 'sentiment_std'] = np.std(scores) if len(scores) > 1 else 0
                    daily_sentiment.loc[date, 'news_volume'] = len(titles)

            # Fill missing values and calculate features
            daily_sentiment = daily_sentiment.fillna(method='ffill', limit=7).fillna(0)

            logging.info(f"Processed news data from {daily_sentiment.index.min()} to {daily_sentiment.index.max()}")

            return daily_sentiment

        except Exception as e:
            logging.error(f"Error processing news data: {str(e)}")
            raise

    def combine_and_normalize_data(self):
        try:
            market_data = self.load_market_data()
            returns_data = self.calculate_returns(market_data)
            news_features = self.process_news_data()

            # Normalize all dates to UTC and remove timezone info
            market_data.index = market_data.index.tz_convert('UTC').tz_localize(None)
            returns_data.index = returns_data.index.tz_convert('UTC').tz_localize(None)
            if news_features.index.tz is not None:
                news_features.index = news_features.index.tz_convert('UTC').tz_localize(None)

            # Print exact index types and samples
            logging.info("\nIndex type analysis:")
            logging.info(f"Market data index type: {type(market_data.index)}")
            logging.info(f"Returns data index type: {type(returns_data.index)}")
            logging.info(f"News data index type: {type(news_features.index)}")

            # Find overlapping date range automatically
            start_date = max(market_data.index.min(),
                             returns_data.index.min(),
                             news_features.index.min())
            end_date = min(market_data.index.max(),
                           returns_data.index.max(),
                           news_features.index.max())

            logging.info("\nFiltering dates:")
            logging.info(f"Start date: {start_date}")
            logging.info(f"End date: {end_date}")

            # Filter data and check lengths
            market_filtered = market_data[start_date:end_date]
            returns_filtered = returns_data[start_date:end_date]
            news_filtered = news_features[start_date:end_date]

            logging.info("\nFiltered data lengths:")
            logging.info(f"Market data: {len(market_filtered)} rows")
            logging.info(f"Returns data: {len(returns_filtered)} rows")
            logging.info(f"News data: {len(news_filtered)} rows")

            # Check for empty dataframes
            if market_filtered.empty or returns_filtered.empty or news_filtered.empty:
                logging.error("One or more filtered datasets are empty")
                return None

            # Get actual date ranges after filtering
            logging.info("\nFiltered date ranges:")
            logging.info(f"Market data: {market_filtered.index.min()} to {market_filtered.index.max()}")
            logging.info(f"Returns data: {returns_filtered.index.min()} to {returns_filtered.index.max()}")
            logging.info(f"News data: {news_filtered.index.min()} to {news_filtered.index.max()}")

            # Combine filtered data
            combined_data = pd.concat([returns_filtered, news_filtered], axis=1)

            logging.info(f"\nCombined data shape: {combined_data.shape}")
            logging.info(f"Combined date range: {combined_data.index.min()} to {combined_data.index.max()}")

            return combined_data

        except Exception as e:
            logging.error(f"Error in combine_and_normalize_data: {str(e)}")
            logging.error(f"Full traceback: {traceback.format_exc()}")
            raise

    def prepare_training_sequences(self, data, sequence_length=20):
        """Prepare sequences with proper NaN handling"""
        try:
            # Handle missing values first
            data = self._handle_missing_values(data.copy())

            # Log data statistics before normalization
            logging.info("Data statistics before normalization:")
            logging.info(f"Mean: {data.mean().mean():.4f}")
            logging.info(f"Std: {data.std().mean():.4f}")

            # Initialize and fit scaler if not already done
            if self.scaler is None:
                self.scaler = StandardScaler()
                normalized_data = pd.DataFrame(
                    self.scaler.fit_transform(data),
                    columns=data.columns,
                    index=data.index
                )
            else:
                normalized_data = pd.DataFrame(
                    self.scaler.transform(data),
                    columns=data.columns,
                    index=data.index
                )

            # Verify normalization
            logging.info("Data statistics after normalization:")
            logging.info(f"Mean: {normalized_data.mean().mean():.4f}")
            logging.info(f"Std: {normalized_data.std().mean():.4f}")

            # Create sequences
            X = []
            y = []
            returns_columns = [col for col in normalized_data.columns if 'Returns' in col]
            feature_columns = [col for col in normalized_data.columns
                               if col not in returns_columns]

            for i in range(len(normalized_data) - sequence_length):
                X_sequence = normalized_data[feature_columns].iloc[i:i + sequence_length].values
                y_sequence = normalized_data[returns_columns].iloc[i + sequence_length].values

                # Only add sequences with no NaN values
                if not np.isnan(X_sequence).any() and not np.isnan(y_sequence).any():
                    X.append(X_sequence)
                    y.append(y_sequence)

            X = np.array(X)
            y = np.array(y)

            # Final validation
            if len(X) == 0 or len(y) == 0:
                raise ValueError("No valid sequences could be created")

            if np.isnan(X).any():
                raise ValueError(f"NaN values in features: {np.isnan(X).sum()} / {X.size}")
            if np.isnan(y).any():
                raise ValueError(f"NaN values in targets: {np.isnan(y).sum()} / {y.size}")

            logging.info(f"Created {len(X)} valid sequences")
            logging.info(f"X shape: {X.shape}, y shape: {y.shape}")
            logging.info(f"Feature statistics - Mean: {X.mean():.4f}, Std: {X.std():.4f}")
            logging.info(f"Target statistics - Mean: {y.mean():.4f}, Std: {y.std():.4f}")

            return X, y, returns_columns

        except Exception as e:
            logging.error(f"Error in prepare_training_sequences: {str(e)}")
            logging.error(f"Full traceback: {traceback.format_exc()}")
            raise

    def create_train_test_split(self, X, y, train_size=0.7, val_size=0.15):
        """Create train/validation/test split"""
        try:
            n_samples = len(X)
            train_end = int(n_samples * train_size)
            val_end = int(n_samples * (train_size + val_size))

            X_train = X[:train_end]
            y_train = y[:train_end]

            X_val = X[train_end:val_end]
            y_val = y[train_end:val_end]

            X_test = X[val_end:]
            y_test = y[val_end:]

            logging.info(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

            return X_train, X_val, X_test, y_train, y_val, y_test

        except Exception as e:
            logging.error(f"Error in create_train_test_split: {str(e)}")
            raise