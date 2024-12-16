from src.utils.data_processor import DataProcessor
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

        # Separate LSTMs for market and political data
        self.market_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        self.political_lstm = nn.LSTM(
            input_size=5,  # Number of political features
            hidden_size=hidden_size // 2,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanisms
        self.market_attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.political_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Prediction layers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_stocks)
        )

    def forward(self, x):
        # Split input into market and political features
        market_features = x[:, :, :-5]  # All but last 5 features
        political_features = x[:, :, -5:]  # Last 5 features
        
        # Process market data
        market_out, _ = self.market_lstm(market_features)
        market_attention = self.market_attention(market_out)
        market_context = torch.sum(market_attention * market_out, dim=1)
        
        # Process political data
        political_out, _ = self.political_lstm(political_features)
        political_attention = self.political_attention(political_out)
        political_context = torch.sum(political_attention * political_out, dim=1)
        
        # Combine features
        combined = torch.cat([market_context, political_context], dim=1)
        fused = self.fusion(combined)
        
        # Make predictions
        predictions = self.fc_layers(fused)
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

    def train_model(self, X_train, X_test, y_train, y_test, returns_columns):
        """Train the model and return training history"""
        # Initialize model with number of stocks
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        model = LSTM(
            input_size=self.config['input_size'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout'],
            num_stocks=len(returns_columns)  # Number of stocks to predict
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
            # Normalisiere die Aufmerksamkeitsgewichte mit Softmax
            market_attention_weights = torch.softmax(market_attention, dim=1)
            market_context = torch.sum(market_attention_weights * market_out, dim=1)

            political_attention_weights = torch.softmax(political_attention, dim=1)
            political_context = torch.sum(political_attention_weights * political_out, dim=1)
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
    # Define data files
    market_files = ['AAPL_price_history.csv', 'GOOGL_price_history.csv',
                   'MSFT_price_history.csv', 'AMZN_price_history.csv']
    news_file = 'news_data.csv'

    # Process data
    processor = DataProcessor(market_files, news_file)
    processed_data = processor.combine_and_normalize_data()
    X, y, returns_columns = processor.prepare_training_sequences(processed_data, sequence_length=2520)
    X_train, X_test, y_train, y_test = processor.create_train_test_split(X, y, train_ratio=0.8)

    # Enhanced configuration
    config = {
        'input_size': X.shape[2],  # Number of features
        'hidden_size': 512,        # Increased for more capacity
        'num_layers': 4,          # More layers for complex patterns
        'dropout': 0.4,           # Increased to prevent overfitting
        'batch_size': 32,         # Reduced due to larger sequences
        'learning_rate': 0.0001,  # Reduced for stability
        'epochs': 300,            # Increased for better convergence
        'patience': 25,           # Increased patience
        'sequence_length': 2520   # 10 years of trading days
    }

    # Initialize trainer and train model
    trainer = ModelTrainer(config)
    model, history = trainer.train_model(X_train, X_test, y_train, y_test, returns_columns)

    # Evaluate model
    metrics = trainer.evaluate_model(model, X_test, y_test)
    print("\nModel Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.6f}")

if __name__ == "__main__":
    main()