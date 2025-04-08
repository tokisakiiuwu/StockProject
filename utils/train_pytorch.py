import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

def add_lag_features(df, columns, lags=[1, 2, 3]):
    for col in columns:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df

def train_model(df: pd.DataFrame):
    df = df.tail(252).copy()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['P/E'] = df.get('trailingPE', 15)

    df = add_lag_features(df, ['Close', 'Volume', 'P/E', 'EMA50'], lags=[1, 2, 3])
    df['Target'] = df['Close'].shift(-30)
    df.dropna(inplace=True)

    feature_cols = [f"{col}_lag{lag}" for col in ['Close', 'Volume', 'P/E', 'EMA50'] for lag in [1, 2, 3]]
    X = df[feature_cols].values
    y = df['Target'].values.reshape(-1, 1)

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    model = StockLSTM(input_size=len(feature_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(1000):
        model.train()
        output = model(X_tensor)
        loss = loss_fn(output, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1} - Loss: {loss.item():.6f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/lstm_model.pt")
    joblib.dump((scaler_x, scaler_y, feature_cols), "models/scalers.pkl")
    print("✅ Training complete. Model saved.")
