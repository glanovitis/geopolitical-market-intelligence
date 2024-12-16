import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import streamlit as st
import torch
from src.data.data_collector import MarketDataCollector
from src.models.model_trainer import ModelTrainer
from src.utils.data_processor import DataProcessor

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

    # Initialize PyTorch device
    device = check_cuda()

    # Initialize components
    load_dotenv()

    # Define market symbols and other parameters
    financial_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    years_of_history = 10

    try:
        # Initialize collector and collect data
        with st.spinner('Collecting market and news data...'):
            collector = MarketDataCollector(
                financial_symbols=financial_symbols,
                years_of_history=years_of_history
            )
            dataset = collector.prepare_combined_dataset()

            # Create directories if they don't exist
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_dir = os.path.join('data', f'collection_{timestamp}')
            market_dir = os.path.join(base_dir, 'market_data')
            news_dir = os.path.join(base_dir, 'news_data')

            os.makedirs(market_dir, exist_ok=True)
            os.makedirs(news_dir, exist_ok=True)

            # Save the collected data
            collector.save_to_csv(dataset)

            # Wait for files to be saved
            st.success('Data collection completed!')

            # Get paths of saved files
            market_files = [
                os.path.join(root, file)
                for root, _, files in os.walk(market_dir)
                for file in files if file.endswith('.csv')
            ]

            news_files = [
                os.path.join(root, file)
                for root, _, files in os.walk(news_dir)
                for file in files if file.endswith('.csv')
            ]

            if not market_files or not news_files:
                raise FileNotFoundError("No data files were created during collection")

            # Use the first news file found
            news_file = news_files[0]

            # Initialize data processor
            with st.spinner('Processing and combining data...'):
                processor = DataProcessor(market_files, news_file)
                processed_data = processor.combine_and_normalize_data()

                # Prepare sequences for training
                X, y, returns_columns = processor.prepare_training_sequences(
                    processed_data,
                    sequence_length=2520  # 10 years of trading days
                )

                # Move data to appropriate device
                X = torch.FloatTensor(X).to(device)
                y = torch.FloatTensor(y).to(device)

                # Create train-test split
                X_train, X_val, X_test, y_train, y_val, y_test = processor.create_train_test_split(X, y)

            # Model configuration
            config = {
                'input_size': X.shape[2],
                'hidden_size': 512,
                'num_layers': 4,
                'dropout': 0.4,
                'batch_size': 32,
                'learning_rate': 0.0001,
                'epochs': 300,
                'patience': 25,
                'sequence_length': 2520,
                'device': device
            }

            # Initialize and train model
            with st.spinner('Training model...'):
                trainer = ModelTrainer(config)
                model, history = trainer.train_model(X_train, X_test, y_train, y_test, returns_columns)

                # Evaluate model
                metrics = trainer.evaluate_model(model, X_test, y_test)

                st.success('Model training completed!')

                # Display results
                st.subheader("Model Evaluation Metrics")
                for metric, value in metrics.items():
                    st.write(f"{metric.upper()}: {value:.6f}")

                # Display training history plot
                st.subheader("Training History")
                history_path = os.path.join(trainer.model_dir, 'training_history.png')
                if os.path.exists(history_path):
                    st.image(history_path)

                # Display dataset statistics
                st.subheader("Dataset Statistics")
                st.write(f"Total sequences: {len(X)}")
                st.write(f"Training sequences: {len(X_train)}")
                st.write(f"Validation sequences: {len(X_val)}")
                st.write(f"Testing sequences: {len(X_test)}")

                # Display feature importance
                st.subheader("Feature Importance")
                feature_names = processed_data.columns.tolist()
                st.write("Top features used in prediction:")
                st.write(feature_names[:10])  # Display top 10 features

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()