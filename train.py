import json
import logging
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd

from src.data.fetcher import DataFetcher
from src.features.ict_features import ICTFeatures
from src.models.lstm_model import ICTDataset, MultiTaskICTLSTM, ModelTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_synthetic_data(config_path="config.json"):
    logger.info("Starting data generation for training...")
    with open(config_path, 'r') as f:
        config = json.load(f)

    fetcher = DataFetcher(exchange_id=config["exchange"]["name"])
    symbols = config["trading"]["symbols"]
    base_tf = config["trading"]["base_timeframe"]

    all_data = []

    for symbol in symbols:
        logger.info(f"Fetching historical data for {symbol}...")
        df = fetcher.fetch_historical_data(symbol, base_tf, limit=5000)
        if df.empty:
            continue

        logger.info(f"Generating ICT features for {symbol}...")
        features_engine = ICTFeatures(df)
        features_df = features_engine.generate_all_features()
        all_data.append(features_df)

    if not all_data:
        raise ValueError("Failed to fetch historical data for training.")

    combined_df = pd.concat(all_data)
    combined_df.to_csv("data/training_data.csv")
    logger.info("Data generation complete and saved to data/training_data.csv")
    return combined_df

def train_model(config_path="config.json", data_path="data/training_data.csv"):
    with open(config_path, 'r') as f:
        config = json.load(f)

    if not os.path.exists(data_path):
        generate_synthetic_data(config_path)

    logger.info("Loading training data...")
    df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)

    # Init Dataset
    seq_length = 60
    dataset = ICTDataset(df, seq_length=seq_length)
    dataloader = DataLoader(dataset, batch_size=config["model"]["batch_size"], shuffle=True)

    # Init Model
    input_size = len(dataset.feature_cols)
    hidden_size = config["model"]["lstm_hidden_size"]
    num_layers = config["model"]["lstm_num_layers"]

    model = MultiTaskICTLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_tasks=len(dataset.target_cols)
    )

    trainer = ModelTrainer(model, learning_rate=config["model"]["learning_rate"])

    epochs = config["model"]["epochs"]
    logger.info(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        loss = trainer.train_epoch(dataloader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}")

    model_path = config["model"]["model_path"]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    trainer.save_model(model_path)

    # Save the global scaler for inference consistency
    scaler_path = os.path.join(os.path.dirname(model_path), "scaler.json")
    scaler_data = {
        "mean": dataset.data_mean.to_dict(),
        "std": dataset.data_std.to_dict()
    }
    with open(scaler_path, "w") as f:
        json.dump(scaler_data, f)

    logger.info(f"Model saved successfully to {model_path}")
    logger.info(f"Scaler saved successfully to {scaler_path}")

if __name__ == "__main__":
    train_model()