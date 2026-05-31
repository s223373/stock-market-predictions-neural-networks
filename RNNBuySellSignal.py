import yfinance as yf
import pandas as pd
import numpy as np
import ta
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import mplfinance as mpf
import plotly.graph_objects as go
import datetime
from macd_rsi_features import add_macd_lr_features
from feature_engineering import build_features, FEATURES
from model import StockPriceLSTMNetwork, DirectionalLoss

ticker = "AAPL"
df = yf.download(ticker, period='30d', interval="5m")
df.columns = df.columns.get_level_values(0)
df.index = pd.to_datetime(df.index)
df = build_features(df)
y = df[FEATURES].dropna().values.astype(float)


num_features = y.shape[1]

# test_size = 14 

window_sizes = [7, 14, 30, 91, 182, 365]

date_index = df.index.tz_convert('US/Pacific')

last_date = date_index[-1]

future_dates = pd.bdate_range(start=last_date, periods=31)[1:]

train_dates = date_index
train_set = y

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
train_norm = scaler.fit_transform(train_set)
train_norm = torch.FloatTensor(train_norm)

def generate_windows(sequence, window_size):
    windows = []
    L = len(sequence)
    for i in range(L - window_size):
        window = sequence[i:i + window_size]          
        label  = sequence[i + window_size, 0]  
        windows.append((window,label))
    return windows

epochs = 200

window_size = 14

train_data = generate_windows(train_norm, window_size)
torch.manual_seed(101)
model = StockPriceLSTMNetwork(num_features, 64, 1)


criterion = DirectionalLoss(alpha=0.7)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


model.train()
for epoch in range(epochs):
    epoch_loss = 0.0

    for seq, y_train in train_data:
        optimizer.zero_grad()

        model.hidden = (torch.zeros(model.num_layers, 1, model.hidden_size),
                    torch.zeros(model.num_layers, 1, model.hidden_size))

        y_pred = model(seq)
        loss = criterion(y_pred, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_data)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch+1:2} Avg Loss: {avg_loss:.8f}')



now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
torch.save(model.state_dict(), f'StockPriceLSTMNetwork_{now}.pt')

# pred_series = pd.Series(preds_inv, index=test_dates)

# plt.figure(figsize=(12, 4))
# plt.title(f'{ticker} Stock Price')
# plt.ylabel('Price ($)')
# plt.xlabel('Date')
# plt.plot(date_index, y, label='Actual')
# plt.plot(pred_series, label='Predicted', linestyle='--', color='orange')
# plt.xlim(pd.Timestamp('2022-01-01'), test_dates[-1])
# plt.legend()
# plt.show()