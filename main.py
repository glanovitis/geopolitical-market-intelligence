import os
import warnings
import logging
import torch
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import glob
import json
import traceback
from src.utils.data_processor import DataProcessor
from src.utils.model_trainer import ModelTrainer
from src.data.data_collector import MarketDataCollector
from src.models.market_predictor import MarketPredictor

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

def collect_new_data():
    """Collect new market and news data"""
    try:
        financial_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
        years_of_history = 10

        st.info("Initializing data collection...")
        collector = MarketDataCollector(
            financial_symbols=financial_symbols,
            years_of_history=years_of_history
        )

        with st.spinner("Collecting market and news data..."):
            dataset = collector.prepare_combined_dataset()
            collector.save_to_csv(dataset)
            st.success("Data collection completed successfully!")

        # Return the path to the newest collection directory
        data_dir = 'data'
        collection_dirs = sorted([d for d in os.listdir(data_dir)
                                if d.startswith('collection_') and
                                os.path.isdir(os.path.join(data_dir, d))],
                               reverse=True)
        return os.path.join(data_dir, collection_dirs[0])

    except Exception as e:
        st.error(f"Data collection failed: {str(e)}")
        logging.error(f"Data collection error: {str(e)}")
        logging.error(f"Full traceback: {traceback.format_exc()}")
        raise


def make_predictions(model_path, processor, data):
    """Make predictions using the trained model"""
    try:
        predictor = MarketPredictor(model_path, processor)

        with st.spinner("Making predictions..."):
            predictions = predictor.predict(data['market_data'], data['news_data'])

            if predictions is None:
                st.error("Failed to generate predictions")
                return None

            # Display predictions
            st.subheader("Market Predictions")

            # Display normalized predictions
            st.write("Normalized Predictions:")
            norm_pred_df = pd.DataFrame(
                predictions['normalized'],
                columns=[col.replace('Returns_', '') for col in predictor.returns_columns]
            )
            st.dataframe(norm_pred_df.head())

            # Display original scale predictions
            st.write("Original Scale Predictions:")
            for stock, values in predictions['original_scale'].items():
                stock_name = stock.replace('Returns_', '')

                # Determine the appropriate number of periods and frequency
                if '_5d' in stock:
                    freq = '5D'
                    n_periods = 12  # Show ~60 days worth of 5-day predictions
                    title = f"{stock_name} 5-day predicted returns"
                elif '_20d' in stock:
                    freq = '20D'
                    n_periods = 6  # Show ~120 days worth of 20-day predictions
                    title = f"{stock_name} daily predicted returns"
                else:
                    freq = 'D'
                    n_periods = 30  # Show 30 days of daily predictions
                    title = f"{stock_name} daily predicted returns"

                # Create date range starting from today going forward
                chart_data = pd.DataFrame({
                    'Predicted Returns': values[:n_periods]
                }, index=pd.date_range(start=pd.Timestamp.now(), periods=n_periods, freq=freq))

                st.write(title)
                st.line_chart(chart_data)

            # Save predictions
            save_dir = os.path.dirname(model_path)
            norm_pred_path = os.path.join(save_dir, 'normalized_predictions.csv')
            orig_pred_path = os.path.join(save_dir, 'original_scale_predictions.csv')

            # Save normalized predictions
            norm_pred_df.to_csv(norm_pred_path, index=False)

            # Save original scale predictions
            orig_pred_df = pd.DataFrame(predictions['original_scale'])
            orig_pred_df.to_csv(orig_pred_path, index=False)

            st.success(f"Predictions saved to {save_dir}")

            return predictions

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        logging.error(f"Prediction error: {str(e)}")
        logging.error(f"Full traceback: {traceback.format_exc()}")
        return None

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
            'learning_rate': 0.00005,
            'batch_size': 32,
            'num_epochs': 150,
            'hidden_size': 256,
            'num_layers': 3,
            'dropout': 0.2
        }

        # Get data directory
        data_dir = 'data'
        collection_dirs = sorted([d for d in os.listdir(data_dir)
                                  if d.startswith('collection_') and
                                  os.path.isdir(os.path.join(data_dir, d))],
                                 reverse=True)

        # If no data exists, collect new data
        if not collection_dirs:
            st.warning("No existing data found. Starting new data collection...")
            base_dir = collect_new_data()
            if not base_dir:
                st.error("Data collection failed.")
                return
        else:
            # Get the most recent data collection
            latest_collection = collection_dirs[0]
            base_dir = os.path.join(data_dir, latest_collection)


        st.info(f"Using data from collection: {os.path.basename(base_dir)}")

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

            # Make predictions using the trained model
            st.subheader("Making Predictions")
            if st.button("Generate Market Predictions"):
                try:
                    # Create config file needed by MarketPredictor
                    config = {
                        'input_size': X.shape[2],
                        'hidden_size': hyperparameters['hidden_size'],
                        'num_layers': hyperparameters['num_layers'],
                        'dropout': hyperparameters['dropout']
                    }

                    config_path = os.path.join(base_dir, 'config.json')
                    with open(config_path, 'w') as f:
                        json.dump(config, f)

                    # Prepare data for predictions
                    prediction_data = {
                        'market_data': market_files,
                        'news_data': news_files[-1]  # Use the most recent news data
                    }

                    # Make predictions
                    predictions = make_predictions(model_path, processor, prediction_data)

                    # Save predictions
                    predictions_path = os.path.join(base_dir, 'predictions.csv')
                    pred_df = pd.DataFrame(
                        predictions['normalized'],
                        columns=[col.replace('Returns_', '') for col in returns_columns]
                    )
                    pred_df.to_csv(predictions_path, index=False)
                    st.success(f"Predictions saved to {predictions_path}")

                except Exception as e:
                    st.error(f"Error making predictions: {str(e)}")
                    logging.error(f"Prediction error: {str(e)}")
                    logging.error(f"Full traceback: {traceback.format_exc()}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Detailed error: {str(e)}")
        logging.error(f"Full traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()