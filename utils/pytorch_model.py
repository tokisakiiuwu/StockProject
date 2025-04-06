import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

def add_lag_features(df, columns, lags=[1, 2, 3]):
    for col in columns:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df

def prepare_features(df):
    df = df.copy()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['P/E'] = df.get('trailingPE', 15)
    df = add_lag_features(df, ['Close', 'Volume', 'P/E', 'EMA50'], lags=[1, 2, 3])
    df.dropna(inplace=True)
    return df

def predict_next_month_price(df: pd.DataFrame):
    if df.shape[0] < 100:
        return None  # not enough data

    df = prepare_features(df.tail(252))

    scaler_x, scaler_y, feature_cols = joblib.load("models/scalers.pkl")

    last_row = df[feature_cols].values[-1:]
    X_scaled = scaler_x.transform(last_row)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)

    model = StockLSTM(input_size=len(feature_cols))
    model.load_state_dict(torch.load("models/lstm_model.pt", map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        pred_scaled = model(X_tensor).item()
        pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]

    return round(pred, 2)
