import streamlit as st
from PIL.PcxImagePlugin import logger
from dotenv import load_dotenv
import os
import warnings
import logging
from datetime import datetime
import traceback
from src.data.data_collector import MarketDataCollector
from src.utils.data_processor import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    st.title("Geopolitical Market Intelligence")

    try:
        load_dotenv()

        # Define parameters
        financial_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
        years_of_history = 1
        sequence_length = 60

        # Setup data directories
        data_dir = 'data'
        os.makedirs(data_dir, exist_ok=True)

        # Find existing data collections
        collection_dirs = sorted([d for d in os.listdir(data_dir)
                                  if d.startswith('collection_') and
                                  os.path.isdir(os.path.join(data_dir, d))],
                                 reverse=True)

        with st.spinner('Processing data...'):
            if not collection_dirs:
                st.info("No existing data found. Starting new data collection...")
                collector = MarketDataCollector(
                    financial_symbols=financial_symbols,
                    years_of_history=years_of_history
                )
                dataset = collector.prepare_combined_dataset()
                collector.save_to_csv(dataset)

                # Refresh collection dirs
                collection_dirs = sorted([d for d in os.listdir(data_dir)
                                          if d.startswith('collection_')],
                                         reverse=True)

            # Get latest collection
            latest_collection = collection_dirs[0]
            base_dir = os.path.join(data_dir, latest_collection)
            market_dir = os.path.join(base_dir, 'market_data')
            news_dir = os.path.join(base_dir, 'news_data')

            st.info(f"Using data from collection: {latest_collection}")

            # Get data files
            market_files = [os.path.join(market_dir, f) for f in os.listdir(market_dir)
                            if f.endswith('.csv')]
            news_files = sorted([os.path.join(news_dir, f) for f in os.listdir(news_dir)
                                 if f.endswith('.csv')])

            if not market_files:
                raise FileNotFoundError(f"No market data files found in {market_dir}")
            if not news_files:
                raise FileNotFoundError(f"No news data files found in {news_dir}")

            st.write(f"Found {len(market_files)} market data files")
            st.write(f"Found {len(news_files)} news data files")

            # Process data
            processor = DataProcessor(market_files, news_files[-1])
            processed_data = processor.combine_and_normalize_data()

            # Prepare sequences
            X, y, returns_columns = processor.prepare_training_sequences(
                processed_data,
                sequence_length=sequence_length
            )

            # Create splits
            X_train, X_val, X_test, y_train, y_val, y_test = processor.create_train_test_split(X, y)

            # Display statistics
            st.subheader("Dataset Statistics")
            st.write(f"Total sequences: {len(X)}")
            st.write(f"Training sequences: {len(X_train)}")
            st.write(f"Validation sequences: {len(X_val)}")
            st.write(f"Testing sequences: {len(X_test)}")
            st.write(f"Number of features: {X.shape[2]}")
            st.write("Return columns:", returns_columns)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Detailed error: {str(e)}")
        logging.error("Full traceback:")
        logging.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()