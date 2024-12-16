import os
import warnings
import logging
import torch
import streamlit as st
from dotenv import load_dotenv
import glob
import traceback
from src.utils.data_processor import DataProcessor
from src.utils.model_trainer import ModelTrainer

# Configure logging and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
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
        device = check_cuda()

        # Define parameters
        financial_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
        years_of_history = 10
        sequence_length = 60  # 60 days of history for prediction

        # Model hyperparameters
        hyperparameters = {
            'learning_rate': 0.0001,  # Reduced from 0.001
            'batch_size': 16,  # Reduced from 32
            'num_epochs': 100,
            'hidden_size': 64,  # Reduced from 128
            'num_layers': 2,
            'dropout': 0.1  # Reduced from 0.2
        }

        # Get data directory
        data_dir = 'data'
        collection_dirs = sorted([d for d in os.listdir(data_dir)
                                  if d.startswith('collection_') and
                                  os.path.isdir(os.path.join(data_dir, d))],
                                 reverse=True)

        if not collection_dirs:
            st.warning("No existing data found. Starting new data collection...")
            return

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

        # Remove cached files to force reprocessing
        cached_data_path = os.path.join(base_dir, 'processed_data.pkl')
        cached_sequences_path = os.path.join(base_dir, f'sequences_len{sequence_length}.npz')

        if os.path.exists(cached_data_path):
            os.remove(cached_data_path)
        if os.path.exists(cached_sequences_path):
            os.remove(cached_sequences_path)

        # Initialize processor
        processor = DataProcessor(market_files, news_files[-1])

        # Load or process data
        with st.spinner("Loading/Processing data..."):
            processed_data = processor.load_or_process_data()
            st.success("Data processing completed!")

        # Load or create sequences
        with st.spinner("Preparing sequences..."):
            X, y, returns_columns = processor.load_or_create_sequences(
                processed_data,
                sequence_length
            )
            st.success("Sequence preparation completed!")

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

        # Initialize and train model
        st.subheader("Model Training")
        with st.spinner("Training model..."):
            trainer = ModelTrainer(
                input_size=X.shape[2],
                output_size=len(returns_columns),
                device=device,
                **hyperparameters
            )

            # Train the model
            training_progress = st.progress(0)
            history = trainer.train(
                X_train, y_train,
                X_val, y_val,
                progress_bar=training_progress
            )

            # Plot training history
            st.subheader("Training History")
            trainer.plot_training_history(history)

            # Evaluate model
            st.subheader("Model Evaluation")
            test_loss, test_metrics = trainer.evaluate(X_test, y_test)

            st.write("Test Results:")
            st.write(f"Loss: {test_loss:.4f}")
            for metric_name, value in test_metrics.items():
                st.write(f"{metric_name}: {value:.4f}")

            # Save the model
            model_path = os.path.join(base_dir, 'trained_model.pth')
            trainer.save_model(model_path)
            st.success(f"Model saved to {model_path}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Detailed error: {str(e)}")
        logging.error(f"Full traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()