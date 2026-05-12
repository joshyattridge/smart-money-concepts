import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple

class ICTDataset(Dataset):
    def __init__(self, features_df: pd.DataFrame, seq_length: int = 60):
        """
        Args:
            features_df: DataFrame with input features (OHLCV + deterministic ICT labels)
            seq_length: Number of time steps in a sequence
        """
        # Feature columns: price, volume, and momentum metrics
        self.feature_cols = ['open', 'high', 'low', 'close', 'volume', 'momentum', 'avg_volume', 'premium_discount']

        # Target columns: the ICT concepts we want the model to predict
        self.target_cols = ['mss', 'fvg', 'ob', 'liquidity_grab', 'breaker_block', 'ote']

        self.seq_length = seq_length

        # Normalize continuous features
        self.data_mean = features_df[self.feature_cols].mean()
        self.data_std = features_df[self.feature_cols].std()
        normalized_features = (features_df[self.feature_cols] - self.data_mean) / (self.data_std + 1e-8)

        self.X = normalized_features.values

        # Targets are categorical: -1 (bearish), 0 (none), 1 (bullish)
        # For cross entropy, we map [-1, 0, 1] to [0, 1, 2]
        targets_raw = features_df[self.target_cols].values
        self.y = targets_raw + 1

    def __len__(self):
        return len(self.X) - self.seq_length

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        x_seq = self.X[idx:idx + self.seq_length]
        # Target is the ICT concepts at the end of the sequence
        y_val = self.y[idx + self.seq_length - 1]

        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)

class MultiTaskICTLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_tasks: int = 6, classes_per_task: int = 3):
        super(MultiTaskICTLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_tasks = num_tasks

        # Shared LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)

        # Task-specific prediction heads
        # Each task predicts 3 classes: 0 (bearish), 1 (none), 2 (bullish)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, classes_per_task)
            ) for _ in range(num_tasks)
        ])

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        # Take the output from the last time step
        last_out = out[:, -1, :]

        # Pass through each task-specific head
        task_outputs = [head(last_out) for head in self.heads]
        return task_outputs

class ModelTrainer:
    def __init__(self, model: MultiTaskICTLSTM, learning_rate: float = 0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_x)

            # Calculate loss across all tasks
            loss = 0.0
            for i in range(self.model.num_tasks):
                task_target = batch_y[:, i]
                loss += self.criterion(outputs[i], task_target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        """
        Predicts ICT concepts for a single sequence.
        Returns array of predictions mapped back to [-1, 0, 1].
        """
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
            outputs = self.model(x_tensor)

            predictions = []
            for out in outputs:
                # Get the predicted class (0, 1, or 2)
                pred_class = torch.argmax(out, dim=1).item()
                # Map back to [-1, 0, 1]
                predictions.append(pred_class - 1)

        return np.array(predictions)

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
