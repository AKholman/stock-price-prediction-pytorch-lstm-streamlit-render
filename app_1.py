import streamlit as st
import yfinance as yf
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ---------------------------
# 1️⃣ Define the same LSTM architecture used in training
# ---------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # last time step output
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------------------------
# 2️⃣ Streamlit UI
# ---------------------------
st.title("📈 AAPL Next-Day Stock Price Prediction (LSTM - PyTorch)")
st.write("This app predicts the **next-day adjusted close price** of Apple (AAPL) using a pre-trained PyTorch LSTM model.")

# ---------------------------
# 3️⃣ Load latest Yahoo Finance data
# ---------------------------
with st.spinner("Downloading latest stock data..."):
    df = yf.download("AAPL", period="5y")
st.write("Recent data:", df.tail())

# ---------------------------
# 4️⃣ Prepare data (same preprocessing as training)
# ---------------------------
features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

time_steps = 60
last_60_days = scaled_data[-time_steps:]
X_input = np.expand_dims(last_60_days, axis=0)
X_input = torch.tensor(X_input, dtype=torch.float32)

# ---------------------------
# 5️⃣ Load the trained model
# ---------------------------
input_size = len(features)
model = LSTMModel(input_size)
model.load_state_dict(torch.load("best_lstm_model.pth", map_location=torch.device('cpu')))
model.eval()

# ---------------------------
# 6️⃣ Make prediction
# ---------------------------
with torch.no_grad():
    next_day_scaled = model(X_input).numpy()

# To inverse transform, build a dummy row
dummy = np.zeros((1, len(features)))
dummy[0, 4] = next_day_scaled[0, 0]  # place prediction in 'Adj Close'
next_day_pred = scaler.inverse_transform(dummy)[0, 4]

st.subheader(f"Predicted Next-Day Adj Close: **${next_day_pred:.2f}**")

# ---------------------------
# 7️⃣ Plot results
# ---------------------------
st.subheader("📊 Last 100 Days + Predicted Point")

plt.figure(figsize=(10, 5))
plt.plot(df['Adj Close'][-100:], label='Actual')
plt.scatter(len(df), next_day_pred, color='red', label='Predicted Next Day')
plt.legend()
plt.title("AAPL Stock Price Prediction (Next Day)")
plt.xlabel("Days")
plt.ylabel("Price ($)")
st.pyplot(plt)
