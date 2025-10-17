import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import datetime

# ------------------------------
# 1️⃣ Define the LSTM model class
# ------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# ------------------------------
# 2️⃣ Load trained model
# ------------------------------
input_size = 6
hidden_size = 50
num_layers = 2
output_size = 1

model = LSTMModel(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load("best_lstm_model.pth", map_location=torch.device("cpu")))
model.eval()

# ------------------------------
# 3️⃣ Streamlit UI
# ------------------------------
st.title("📈 AAPL Next-Day Stock Price Prediction (Pytorch-LSTM Model)")

# ------------------------------
# 4️⃣ Fetch Yahoo Finance data safely
# ------------------------------
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=365 * 5)

try:
    df = yf.download("AAPL", start=start_date, end=end_date)
except Exception as e:
    st.error(f"❌ Error downloading data: {e}")
    st.stop()

if df is None or df.empty:
    st.error("⚠️ No data downloaded. Try reloading the app in a few seconds.")
    st.stop()

st.write("Recent Data:", df.tail())

# ------------------------------
# 5️⃣ Feature scaling
# ------------------------------
features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

# ensure all required columns exist
for f in features:
    if f not in df.columns:
        st.error(f"⚠️ Missing column in data: {f}")
        st.stop()

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

# ------------------------------
# 6️⃣ Prepare last 60 days
# ------------------------------
time_steps = 60
if len(scaled_data) < time_steps:
    st.error("⚠️ Not enough data to make prediction.")
    st.stop()

last_60 = scaled_data[-time_steps:]
x_input = np.expand_dims(last_60, axis=0)
x_tensor = torch.tensor(x_input, dtype=torch.float32)

# ------------------------------
# 7️⃣ Predict
# ------------------------------
with torch.no_grad():
    pred_scaled = model(x_tensor).numpy()

# Reconstruct shape for inverse transform
zero_pad = np.zeros((1, 5))
pred_combined = np.concatenate((zero_pad, pred_scaled), axis=1)
next_day_pred = scaler.inverse_transform(pred_combined)[:, 4]  # Adj Close

# ------------------------------
# 8️⃣ Display result
# ------------------------------
st.subheader(f"Predicted Next-Day Adj Close: **${next_day_pred[0]:.2f}**")

# ------------------------------
# 9️⃣ Plot
# ------------------------------
st.subheader("📊 Price Trend (Last 100 days + Prediction)")
plt.figure(figsize=(10, 5))
plt.plot(df['Adj Close'][-100:], label='Actual')
plt.scatter(len(df) - 1, next_day_pred, color='red', label='Predicted Next Day')
plt.legend()
st.pyplot(plt)