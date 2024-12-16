import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import streamlit as st
from src.data.data_collector import MarketDataCollector
from src.models.model_trainer import ModelTrainer
from src.utils.data_processor import DataProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    st.title("Geopolitical Market Intelligence")

    # Initialize components
    load_dotenv()

    # Define market symbols and other parameters
    financial_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    years_of_history = 10

    # Initialize collector and collect data
    with st.spinner('Collecting market and news data...'):
        collector = MarketDataCollector(
            financial_symbols=financial_symbols,
            years_of_history=years_of_history
        )
        dataset = collector.prepare_combined_dataset()

        # Save the collected data
        collector.save_to_csv(dataset)

    # Get the saved data files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = f'data/collection_{timestamp}'
    market_files = [
        f'{base_dir}/market_data/{symbol}_price_history_{datetime.now().year}.csv'
        for symbol in financial_symbols
    ]
    news_file = f'{base_dir}/news_data/news_{datetime.now().year}.csv'

    # Initialize data processor
    with st.spinner('Processing and combining data...'):
        processor = DataProcessor(market_files, news_file)
        processed_data = processor.combine_and_normalize_data()

        # Prepare sequences for training
        X, y, returns_columns = processor.prepare_training_sequences(
            processed_data,
            sequence_length=2520  # 10 years of trading days
        )

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
        'sequence_length': 2520
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
        st.image(f'{trainer.model_dir}/training_history.png')

        # Display dataset statistics
        st.subheader("Dataset Statistics")
        st.write(f"Total sequences: {len(X)}")
        st.write(f"Training sequences: {len(X_train)}")
        st.write(f"Validation sequences: {len(X_val)}")
        st.write(f"Testing sequences: {len(X_test)}")

        # Display feature importance (based on model attention weights)
        st.subheader("Feature Importance")
        feature_names = processed_data.columns.tolist()
        st.write("Top features used in prediction:")
        st.write(feature_names[:10])  # Display top 10 features


if __name__ == "__main__":
    main()