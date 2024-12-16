import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from transformers import pipeline
from datetime import datetime
import torch
from collections import defaultdict
from statsmodels.tsa.seasonal import seasonal_decompose
import logging

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
        """Load and combine market data from multiple files with 10-year history"""
        market_dfs = {}
        for file in self.market_data_files:
            try:
                # Extract symbol from filename
                symbol = os.path.basename(file).split('_')[0]

                # Read the CSV file
                df = pd.read_csv(file)

                # Print debug information
                print(f"Loading data for {symbol}")
                print(f"Columns in file: {df.columns.tolist()}")
                print(f"First few rows of Date column: {df['Date'].head()}")

                # Convert to datetime with explicit format and error handling
                try:
                    # First try parsing as is
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                except Exception as e:
                    print(f"First datetime conversion attempt failed: {e}")
                    try:
                        # Try parsing with explicit format if needed
                        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
                    except Exception as e:
                        print(f"Second datetime conversion attempt failed: {e}")
                        raise

                # Drop any rows where date conversion failed
                df = df.dropna(subset=['Date'])

                if df.empty:
                    raise ValueError(f"No valid data remaining for {symbol} after date processing")

                # Ensure timezone naive
                if hasattr(df['Date'].dt, 'tz_localize'):
                    df['Date'] = df['Date'].dt.tz_localize(None)

                # Filter for last 10 years
                ten_years_ago = pd.Timestamp.now() - pd.DateOffset(years=10)
                df = df[df['Date'] >= ten_years_ago]

                # Set index and convert numeric columns
                df.set_index('Date', inplace=True)
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                df[numeric_columns] = df[numeric_columns].astype('float64')

                market_dfs[symbol] = df
                print(f"Successfully processed {symbol} data")

            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue

        if not market_dfs:
            raise ValueError("No market data could be processed")

        # Combine all market data
        combined_market = pd.concat(market_dfs.values(), axis=1, keys=market_dfs.keys())
        print(f"Loaded market data from {combined_market.index.min()} to {combined_market.index.max()}")
        return combined_market

    def process_market_features(self, market_data):
        if market_data is None or market_data.empty:
            raise ValueError("Market data cannot be empty")
    
        processed_data = market_data.copy()
    
        # Validate data types
        if not all(processed_data.dtypes == np.float64):
            raise ValueError("All market data must be float64")
    
        # Handle missing values more carefully
        missing_count = processed_data.isnull().sum()
        if missing_count.any():
            logger.warning(f"Missing values detected: {missing_count}")
        
        # Use forward fill then backward fill instead of zero
        return processed_data.ffill().bfill()

    def process_news_data(self):
        """Process 10 years of news data with enhanced political sentiment analysis"""
        try:
            news_df = pd.read_csv(self.news_data_file)

            # Print the data types and first few rows for debugging
            print("News data types:", news_df.dtypes)
            print("First few rows of news data:", news_df.head())

            # First ensure the published_at column exists
            if 'published_at' not in news_df.columns:
                # Check if 'webPublicationDate' exists (Guardian API uses this name)
                if 'webPublicationDate' in news_df.columns:
                    news_df['published_at'] = news_df['webPublicationDate']
                else:
                    raise ValueError("No publication date column found in news data")

            # Convert to datetime, handling potential format issues
            news_df['published_at'] = pd.to_datetime(news_df['published_at'], errors='coerce')

            # Drop rows where conversion failed
            news_df = news_df.dropna(subset=['published_at'])

            # Make timezone naive
            if hasattr(news_df['published_at'].dt, 'tz_localize'):
                news_df['published_at'] = news_df['published_at'].dt.tz_localize(None)

            # Filter for last 10 years
            ten_years_ago = pd.Timestamp.now() - pd.DateOffset(years=10)
            news_df = news_df[news_df['published_at'] >= ten_years_ago]

            if news_df.empty:
                raise ValueError("No news data found within the last 10 years")

            # Create political sentiment analyzer with specific focus
            political_topics = [
                'policy', 'regulation', 'government', 'election', 'trade war',
                'sanctions', 'federal reserve', 'interest rates', 'legislation',
                'geopolitical', 'conflict', 'war', 'political crisis'
            ]

            # Calculate daily sentiments with political context
            sentiments = defaultdict(lambda: defaultdict(list))

            for _, row in news_df.iterrows():
                try:
                    date = row['published_at'].date()
                    # Check if title and description exist, use empty string if not
                    title = str(row.get('title', ''))
                    description = str(row.get('description', ''))
                    text = f"{title} {description}"

                    # Get general sentiment
                    sentiment = self.sentiment_analyzer(text)[0]
                    base_score = sentiment['score'] if sentiment['label'] == 'POSITIVE' else -sentiment['score']

                    # Analyze political impact
                    political_impact = sum(1 for topic in political_topics if topic.lower() in text.lower())

                    # Categorize sentiment
                    sentiments[date]['general'].append(base_score)
                    sentiments[date]['political_impact'].append(political_impact)
                    sentiments[date]['texts'].append(text)

                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue

            if not sentiments:
                raise ValueError("No valid sentiment data could be processed")

            # Create daily sentiment features
            daily_features = {}
            for date, scores in sentiments.items():
                if scores['general']:  # Only process if we have sentiment scores
                    daily_features[date] = {
                        'sentiment_mean': np.mean(scores['general']),
                        'sentiment_std': np.std(scores['general']) if len(scores['general']) > 1 else 0,
                        'political_impact_score': np.mean(scores['political_impact']),
                        'news_volume': len(scores['general'])
                    }

            # Convert to DataFrame
            sentiment_df = pd.DataFrame.from_dict(daily_features, orient='index')

            if sentiment_df.empty:
                raise ValueError("No sentiment data was generated")

            # Ensure index is datetime
            sentiment_df.index = pd.to_datetime(sentiment_df.index)

            # Fill missing dates with forward fill then backward fill
            full_date_range = pd.date_range(
                start=sentiment_df.index.min(),
                end=sentiment_df.index.max(),
                freq='B'  # Business days
            )
            sentiment_df = sentiment_df.reindex(full_date_range)
            sentiment_df = sentiment_df.fillna(method='ffill').fillna(method='bfill')

            return sentiment_df

        except Exception as e:
            print(f"Error in process_news_data: {e}")
            print(f"News file path: {self.news_data_file}")
            raise

    def combine_and_normalize_data(self):
        """Combine 10 years of market and news data with enhanced features"""
        # Load and process market data
        market_data = self.load_market_data()
        processed_market = self.process_market_features(market_data)
        
        # Process news data
        news_sentiment = self.process_news_data()
        
        # Additional political-economic features
        news_sentiment['sentiment_volatility'] = news_sentiment['sentiment_std'].rolling(window=30).mean()
        news_sentiment['political_impact_ma60'] = news_sentiment['political_impact_score'].rolling(window=60).mean()
        
        # Create interaction features between market and news data
        processed_market_flat = processed_market.copy()
        processed_market_flat.columns = ['_'.join(col).strip() for col in processed_market_flat.columns.values]
        
        # Combine market and news data
        combined_data = processed_market_flat.merge(news_sentiment,
                                                  left_index=True,
                                                  right_index=True,
                                                  how='left')
        
        # Create interaction features
        for symbol in set(col.split('_')[0] for col in processed_market_flat.columns if '_Returns' in col):
            returns_col = f"{symbol}_Returns"
            if returns_col in combined_data.columns:
                # Sentiment-return interactions
                combined_data[f"{symbol}_sentiment_return_interaction"] = (
                    combined_data[returns_col] * combined_data['sentiment_mean']
                )
                
                # Political impact-return interactions
                combined_data[f"{symbol}_political_return_interaction"] = (
                    combined_data[returns_col] * combined_data['political_impact_score']
                )
        
        # Fill missing values
        combined_data = combined_data.fillna(method='ffill').fillna(method='bfill')
        
        # Normalize numerical features
        numerical_columns = combined_data.select_dtypes(include=[np.number]).columns
        combined_data[numerical_columns] = self.market_scaler.fit_transform(combined_data[numerical_columns])
        
        print("Final combined features:", combined_data.columns.tolist())
        return combined_data

    def prepare_training_sequences(self, data, sequence_length=2520):  # 2520 trading days = 10 years (252 trading days per year)
        """
        Prepare sequential data for training with 10-year historical context
        """
        sequences = []
        targets = []
        
        # Find all Returns columns
        returns_columns = [col for col in data.columns if 'Returns' in col]
        if not returns_columns:
            raise ValueError("No Returns columns found in the data")

        # Add long-term technical indicators
        for symbol in set(col.split('_')[0] for col in returns_columns):
            close_col = f"{symbol}_Close"
            
            if close_col in data.columns:
                # Add long-term moving averages
                data[f"{symbol}_MA50"] = data[close_col].rolling(window=50).mean()
                data[f"{symbol}_MA200"] = data[close_col].rolling(window=200).mean()
                data[f"{symbol}_MA500"] = data[close_col].rolling(window=500).mean()
                
                # Long-term trend indicators
                data[f"{symbol}_YearlyReturn"] = data[close_col].pct_change(periods=252)
                data[f"{symbol}_YearlyVolatility"] = data[f"{symbol}_Returns"].rolling(window=252).std()
                
                # Long-term cycle indicators
                data[f"{symbol}_2YearCycle"] = data[close_col].pct_change(periods=504)
                data[f"{symbol}_5YearCycle"] = data[close_col].pct_change(periods=1260)
                
                # Add seasonal decomposition
                try:
                    decomposition = seasonal_decompose(data[close_col], period=252)
                    data[f"{symbol}_Trend"] = decomposition.trend
                    data[f"{symbol}_Seasonal"] = decomposition.seasonal
                    data[f"{symbol}_Residual"] = decomposition.resid
                except:
                    print(f"Could not perform seasonal decomposition for {symbol}")

        print(f"Predicting returns for: {returns_columns}")
        print(f"Total features used: {len(data.columns)}")
        
        # Use stride to reduce memory usage
        stride = 5  # Create sequences every 5 days instead of every day
        
        for i in range(0, len(data) - sequence_length, stride):
            sequence = data.iloc[i:i + sequence_length]
            target = data.iloc[i + sequence_length][returns_columns].values
            
            if not sequence.isnull().any().any():  # Only add complete sequences
                sequences.append(sequence.values)
                targets.append(target)

        # Convert to PyTorch tensors
        X = torch.FloatTensor(np.array(sequences))
        y = torch.FloatTensor(np.array(targets))

        print(f"Created {len(sequences)} sequences of length {sequence_length}")
        return X, y, returns_columns

    def create_train_test_split(self, X, y, train_ratio=0.8, val_ratio=0.1):
        """Split data into training, validation, and testing sets"""
        total_samples = len(X)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        return X_train, X_val, X_test, y_train, y_val, y_test