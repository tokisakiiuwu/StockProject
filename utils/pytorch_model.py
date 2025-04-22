import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib

# LSTM class model
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

# adding the lag features  function
def add_lag_features(df, columns, lags=[1, 2, 3]):
    for col in columns:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df

# getting the features and lagging them function
def prepare_features(df):
    df = df.copy()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['P/E'] = df.get('trailingPE', 15)
    df = add_lag_features(df, ['Close', 'Volume', 'P/E', 'EMA50'], lags=[1, 2, 3])
    df.dropna(inplace=True)
    return df

# loading the model with the data to predict next month's stocks
def predict_next_month_price(df: pd.DataFrame):
    if df.shape[0] < 100:
        return None  # not enough data

    df = prepare_features(df.tail(252))


    scaler_x, scaler_y, feature_cols = joblib.load("models/scalers.pkl") # trained model loaded

    # data needs to be scaled so that it predicts better
    last_row = df[feature_cols].values[-1:]
    X_scaled = scaler_x.transform(last_row)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)

    model = StockLSTM(input_size=len(feature_cols))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("models/lstm_model.pt", map_location=device))
    model.to(device)
    model.eval()

    # Unscale the predicted data using the previous scalers
    with torch.no_grad():
        pred_scaled = model(X_tensor).item()
        pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]

    return round(pred, 2)
