import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        out = self.dropout(last_time_step)
        out = self.fc(out)
        return out


class ModelTrainer:
    def __init__(self, input_size, output_size, hidden_size=128, num_layers=2,
                 dropout=0.2, learning_rate=0.0001, batch_size=32, num_epochs=100,
                 device=None):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate  # Reduced learning rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        self.model = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        logging.info(f"Model initialized on device: {self.device}")
        logging.info(f"Model architecture:\n{self.model}")

    def prepare_data_loader(self, X, y):
        """Convert numpy arrays to DataLoader with detailed validation"""
        # Detailed input validation
        if np.isnan(X).any():
            nan_locations = np.where(np.isnan(X))
            sample_nans = list(zip(nan_locations[0][:5], nan_locations[1][:5], nan_locations[2][:5]))
            logging.error(f"NaN values found in input features at locations (first 5): {sample_nans}")
            logging.error(f"Input shape: {X.shape}")
            logging.error(f"Total NaN count: {np.isnan(X).sum()}")
            raise ValueError("Input features contain NaN values")

        if np.isnan(y).any():
            nan_locations = np.where(np.isnan(y))
            sample_nans = list(zip(nan_locations[0][:5], nan_locations[1][:5]))
            logging.error(f"NaN values found in target values at locations (first 5): {sample_nans}")
            logging.error(f"Target shape: {y.shape}")
            logging.error(f"Total NaN count: {np.isnan(y).sum()}")
            raise ValueError("Target values contain NaN values")

        # Log data statistics
        logging.info(f"Input features - Shape: {X.shape}")
        logging.info(f"Input features - Mean: {np.mean(X):.4f}, Std: {np.std(X):.4f}")
        logging.info(f"Input features - Min: {np.min(X):.4f}, Max: {np.max(X):.4f}")
        logging.info(f"Target values - Shape: {y.shape}")
        logging.info(f"Target values - Mean: {np.mean(y):.4f}, Std: {np.std(y):.4f}")
        logging.info(f"Target values - Min: {np.min(y):.4f}, Max: {np.max(y):.4f}")

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(self, X_train, y_train, X_val, y_val, progress_bar=None):
        """Train the model with gradient clipping and improved monitoring"""
        train_loader = self.prepare_data_loader(X_train, y_train)
        val_loader = self.prepare_data_loader(X_val, y_val)

        history = {
            'train_loss': [],
            'val_loss': []
        }

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        max_grad_norm = 1.0  # Gradient clipping threshold

        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_losses = []

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()

                # Forward pass with gradient checking
                outputs = self.model(batch_X)
                if torch.isnan(outputs).any():
                    logging.error("NaN values detected in model outputs")
                    raise ValueError("Model produced NaN outputs")

                loss = self.criterion(outputs, batch_y)

                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                # Log gradient norms
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                if epoch == 0:
                    logging.info(f"Gradient norm: {total_norm:.4f}")

                self.optimizer.step()
                train_losses.append(loss.item())

            # Validation phase
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    val_loss = self.criterion(outputs, batch_y)
                    val_losses.append(val_loss.item())

            # Calculate average losses
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)

            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)

            # Update progress bar if provided
            if progress_bar is not None:
                progress = (epoch + 1) / self.num_epochs
                progress_bar.progress(progress)

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': best_val_loss,
                }, 'best_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logging.info(f"Early stopping triggered at epoch {epoch + 1}")
                # Load best model
                checkpoint = torch.load('best_model.pth')
                self.model.load_state_dict(checkpoint['model_state_dict'])
                break

            logging.info(f"Epoch {epoch + 1}/{self.num_epochs} - "
                         f"Train Loss: {avg_train_loss:.6f}, "
                         f"Val Loss: {avg_val_loss:.6f}, "
                         f"LR: {self.learning_rate}")

        return history

    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data"""
        test_loader = self.prepare_data_loader(X_test, y_test)

        self.model.eval()
        test_losses = []
        predictions = []
        actuals = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                test_losses.append(loss.item())

                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Calculate metrics
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))

        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': np.sqrt(mse)
        }

        return np.mean(test_losses), metrics

    def plot_training_history(self, history):
        """Plot training and validation loss"""
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        return plt

    def save_model(self, path):
        """Save the model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'output_size': self.output_size,
                'dropout': self.dropout
            }
        }, path)
        logging.info(f"Model saved to {path}")

    def load_model(self, path):
        """Load the model from disk"""
        checkpoint = torch.load(path)

        # Recreate the model with the saved configuration
        self.model = LSTM(
            input_size=checkpoint['model_config']['input_size'],
            hidden_size=checkpoint['model_config']['hidden_size'],
            num_layers=checkpoint['model_config']['num_layers'],
            output_size=checkpoint['model_config']['output_size'],
            dropout=checkpoint['model_config']['dropout']
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f"Model loaded from {path}")