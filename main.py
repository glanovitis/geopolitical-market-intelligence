import streamlit as st
from dotenv import load_dotenv
import os
import warnings
import logging
from datetime import datetime
from src.data.data_collector import MarketDataCollector
from src.utils.data_processor import DataProcessor
import glob
import traceback

# Configure logging and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
logging.basicConfig(level=logging.INFO)
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_cuda():
    """Check CUDA availability and setup"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available, using CPU")
    return device


def main():
    st.title("Geopolitical Market Intelligence")

    try:
        load_dotenv()

        # Define parameters
        financial_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
        years_of_history = 10
        sequence_length = 60  # 60 days of history for prediction

        # Get data directory
        data_dir = 'data'
        collection_dirs = sorted([d for d in os.listdir(data_dir)
                                  if d.startswith('collection_') and
                                  os.path.isdir(os.path.join(data_dir, d))],
                                 reverse=True)

        if not collection_dirs:
            st.warning("No existing data found. Starting new data collection...")
            # Collect new data...

        # Get the most recent data collection
        latest_collection = collection_dirs[0]
        base_dir = os.path.join(data_dir, latest_collection)

        st.info(f"Using data from collection: {latest_collection}")

        # Get market and news files
        market_files = glob.glob(os.path.join(base_dir, 'market_data', '*.csv'))
        news_files = sorted(glob.glob(os.path.join(base_dir, 'news_data', '*.csv')))

        if not market_files or not news_files:
            raise FileNotFoundError("Missing required data files")

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

        # Display dataset information
        st.subheader("Dataset Statistics")
        st.write(f"Total sequences: {len(X)}")
        st.write(f"Sequence length: {sequence_length} days")
        st.write(f"Number of features: {X.shape[2]}")
        st.write(f"Target variables: {returns_columns}")

        # Create train-test split
        X_train, X_val, X_test, y_train, y_val, y_test = processor.create_train_test_split(X, y)

        st.write(f"Training sequences: {len(X_train)}")
        st.write(f"Validation sequences: {len(X_val)}")
        st.write(f"Testing sequences: {len(X_test)}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Detailed error: {str(e)}")
        if hasattr(e, '__traceback__'):
            logging.error("Full traceback:")
            traceback.print_tb(e.__traceback__)
        raise


if __name__ == "__main__":
    main()