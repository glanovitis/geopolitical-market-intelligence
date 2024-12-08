# model_trainer.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json

class MarketDataset(Dataset):
    """Custom Dataset for market data sequences"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_stocks):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        # Output layer now predicts returns for all stocks
        self.fc = nn.Linear(hidden_size, num_stocks)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take only the last time step output
        last_time_step = lstm_out[:, -1, :]
        out = self.dropout(last_time_step)
        predictions = self.fc(out)
        return predictions


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create model directory
        self.model_dir = os.path.join('models', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.model_dir, exist_ok=True)

        # Save config file
        self.save_config()

    def save_config(self):
        """Save configuration to JSON file"""
        config_path = os.path.join(self.model_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

    def create_dataloaders(self, X_train, X_test, y_train, y_test):
        """Create training and testing DataLoaders"""
        train_dataset = MarketDataset(X_train, y_train)
        test_dataset = MarketDataset(X_test, y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )

        return train_loader, test_loader

    def train_model(self, X_train, X_test, y_train, y_test):
        """Train the model and return training history"""
        # Initialize model
        model = LSTM(
            input_size=self.config['input_size'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        ).to(self.device)

        # Create dataloaders
        train_loader, test_loader = self.create_dataloaders(X_train, X_test, y_train, y_test)

        # Initialize optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        criterion = nn.MSELoss()

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': float('inf')
        }

        # Early stopping parameters
        patience = self.config['patience']
        patience_counter = 0

        # Training loop
        for epoch in range(self.config['epochs']):
            # Training phase
            model.train()
            train_losses = []

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            # Validation phase
            model.eval()
            val_losses = []

            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    val_loss = criterion(outputs, batch_y)
                    val_losses.append(val_loss.item())

            # Calculate average losses
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)

            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)

            # Print progress
            print(f'Epoch [{epoch + 1}/{self.config["epochs"]}] - '
                  f'Train Loss: {avg_train_loss:.6f}, '
                  f'Val Loss: {avg_val_loss:.6f}')

            # Early stopping check
            if avg_val_loss < history['best_val_loss']:
                history['best_val_loss'] = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_val_loss,
                }, os.path.join(self.model_dir, 'best_model.pth'))
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

        self.plot_training_history(history)
        return model, history

    def plot_training_history(self, history):
        """Plot training and validation loss history"""
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.model_dir, 'training_history.png'))
        plt.close()

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the model and return metrics"""
        model.eval()
        test_dataset = MarketDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        with torch.no_grad():
            X, y_true = next(iter(test_loader))
            X, y_true = X.to(self.device), y_true.to(self.device)
            y_pred = model(X)

            # Convert to numpy for evaluation
            y_true = y_true.cpu().numpy()
            y_pred = y_pred.cpu().numpy()

            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)

            metrics = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            }

            return metrics


# Example usage
def main():
    # Configuration
    config = {
        'input_size': 53,  # Number of features
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 100,
        'patience': 10
    }

    # Initialize trainer
    trainer = ModelTrainer(config)

    # Assuming you have your data prepared from data_processor.py
    from data_processor import DataProcessor

    # Initialize and prepare data
    market_files = ['AAPL_price_history.csv', 'GOOGL_price_history.csv',
                    'MSFT_price_history.csv', 'AMZN_price_history.csv']
    news_file = 'news_data.csv'

    processor = DataProcessor(market_files, news_file)
    processed_data = processor.combine_and_normalize_data()
    X, y = processor.prepare_training_sequences(processed_data)
    X_train, X_test, y_train, y_test = processor.create_train_test_split(X, y)

    # Train model
    model, history = trainer.train_model(X_train, X_test, y_train, y_test)

    # Evaluate model
    metrics = trainer.evaluate_model(model, X_test, y_test)
    print("\nModel Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.6f}")


if __name__ == "__main__":
    main()