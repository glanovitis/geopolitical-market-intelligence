import torch
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.model_trainer import LSTM
from src.utils.data_processor import DataProcessor


class MarketPredictor:
    def __init__(self, model_path, processor):
        self.model_path = model_path
        self.processor = processor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load config
        config_path = os.path.join(os.path.dirname(model_path), 'config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Process some data to get number of stocks
        processed_data = self.processor.combine_and_normalize_data()
        self.returns_columns = [col for col in processed_data.columns if 'Returns' in col]
        self.num_stocks = len(self.returns_columns)

        self.model = self.load_model()

    def load_model(self):
        model = LSTM(
            input_size=self.config['input_size'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout'],
            num_stocks=self.num_stocks
        ).to(self.device)

        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def predict(self, market_data, news_data, batch_size=32):
        """
        Make predictions for all stocks using batch processing

        Parameters:
            market_data: Market data input
            news_data: News data input
            batch_size: Number of samples to process at once (default: 32)
        """
        try:
            processed_data = self.processor.combine_and_normalize_data()
            X, _, returns_columns = self.processor.prepare_training_sequences(processed_data)
        
            predictions = []
            # Process in batches to avoid memory issues
            for i in range(0, len(X), batch_size):
                batch = torch.FloatTensor(X[i:i+batch_size]).to(self.device)
                with torch.no_grad():
                    batch_pred = self.model(batch)
                    predictions.append(batch_pred.cpu().numpy())
                
            return np.vstack(predictions)
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Clear cache and retry with smaller batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return self.predict(market_data, news_data, batch_size // 2)
            raise

            # Combine all batch predictions
            normalized_predictions = np.vstack(predictions)

            # Get the scaler from the processor
            scaler = self.processor.market_scaler

            # Prepare for inverse transform
            pred_reshaped = np.zeros((normalized_predictions.shape[0], len(scaler.scale_)))

            # Put predictions in the correct columns
            for i, col in enumerate(returns_columns):
                col_idx = list(processed_data.columns).index(col)
                pred_reshaped[:, col_idx] = normalized_predictions[:, i]

            # Inverse transform
            original_scale_full = scaler.inverse_transform(pred_reshaped)

            # Extract returns for each stock
            results = {}
            for i, col in enumerate(returns_columns):
                col_idx = list(processed_data.columns).index(col)
                results[col] = original_scale_full[:, col_idx]

            return {
                'normalized': normalized_predictions,
                'original_scale': results
            }
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise

    def compare_predictions_with_actual(self, market_data, news_data):
        """Compare predicted returns with actual returns for all stocks"""
        # Get predictions
        predictions = self.predict(market_data, news_data, batch_size=64)

        # Get actual returns from processed data
        processed_data = self.processor.combine_and_normalize_data()

        # Get the dates for the predictions
        dates = processed_data.index[-len(next(iter(predictions['original_scale'].values))):]

        # Create comparison DataFrame for each stock
        comparison_dfs = {}
        metrics = {}

        for stock_col in self.returns_columns:
            stock_name = stock_col.split('_')[0]  # Extract stock symbol
            actual_returns = processed_data[stock_col]
            pred_returns = predictions['original_scale'][stock_col]

            comparison = pd.DataFrame({
                'Date': dates,
                f'Predicted_Returns_{stock_name}': pred_returns,
                f'Actual_Returns_{stock_name}': actual_returns[-len(pred_returns):]
            }).set_index('Date')

            # Calculate metrics for each stock
            mse = np.mean((comparison[f'Actual_Returns_{stock_name}'] -
                           comparison[f'Predicted_Returns_{stock_name}']) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(comparison[f'Actual_Returns_{stock_name}'] -
                                 comparison[f'Predicted_Returns_{stock_name}']))

            comparison_dfs[stock_name] = comparison
            metrics[stock_name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae
            }

        # Create plots
        plt.figure(figsize=(15, 10))
        for i, (stock_name, df) in enumerate(comparison_dfs.items(), 1):
            plt.subplot(2, 2, i)
            plt.plot(df.index, df[f'Actual_Returns_{stock_name}'],
                     label='Actual Returns', color='blue')
            plt.plot(df.index, df[f'Predicted_Returns_{stock_name}'],
                     label='Predicted Returns', color='red')
            plt.title(f'{stock_name} Predicted vs Actual Returns')
            plt.xlabel('Date')
            plt.ylabel('Returns (%)')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.savefig('returns_comparison_all_stocks.png')
        plt.close()

        return {
            'comparison_dfs': comparison_dfs,
            'metrics': metrics
        }


def test_predictions():
    """Test function with comparison analysis for all stocks"""
    market_files = ['AAPL_price_history.csv', 'GOOGL_price_history.csv',
                    'MSFT_price_history.csv', 'AMZN_price_history.csv']
    news_file = 'news_data.csv'
    processor = DataProcessor(market_files, news_file)

    # Find latest model
    models_dir = 'models'
    latest_model_dir = max([d for d in os.listdir(models_dir)
                            if os.path.isdir(os.path.join(models_dir, d))])
    model_path = os.path.join(models_dir, latest_model_dir, 'best_model.pth')

    # Make predictions and compare
    predictor = MarketPredictor(model_path, processor)
    market_data = processor.load_market_data()
    news_data = processor.process_news_data()

    results = predictor.compare_predictions_with_actual(market_data, news_data)

    print("\nPrediction Analysis for All Stocks:")
    print("-" * 50)

    for stock_name, metrics in results['metrics'].items():
        print(f"\n{stock_name} Performance Metrics:")
        print(f"Mean Squared Error: {metrics['MSE']:.8f}")
        print(f"Root Mean Squared Error: {metrics['RMSE']:.8f}")
        print(f"Mean Absolute Error: {metrics['MAE']:.8f}")

        print(f"\nLast 5 days comparison for {stock_name}:")
        print(results['comparison_dfs'][stock_name].tail().to_string())

    print("\nA plot 'returns_comparison_all_stocks.png' has been saved showing the comparisons")

    return results


if __name__ == "__main__":
    results = test_predictions()