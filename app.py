import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import date, timedelta

# ---------------------------
# 1. Streamlit page setup
# ---------------------------
st.set_page_config(page_title="Stock Price Prediction (LSTM)", layout="centered")
st.title("📈 Stock Price Prediction using LSTM (PyTorch)")
st.write("This app downloads live stock data from Yahoo Finance and predicts the next-day closing price using an LSTM model.")

# ---------------------------
# 2. User input
# ---------------------------
ticker = st.text_input("Enter stock ticker symbol:", "AAPL")
period = st.selectbox("Select data period:", ["1y", "3y", "5y"], index=2)
st.write(f"Fetching last {period} of data for **{ticker}**...")

# ---------------------------
# 3. Download and preprocess data
# ---------------------------
try:
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False)
    if df.empty:
        st.error("No data found. Try another ticker.")
        st.stop()

    df = df.rename(columns={'Adj Close': 'Adj_Close'})
    df['Target'] = df['Adj_Close'].shift(-1)
    df.dropna(inplace=True)

    features = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X = scaler_X.fit_transform(df[features])
    y = scaler_y.fit_transform(df[['Target']])

    # Create sequences
    def create_sequences(X, y, time_steps=60):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X[i:(i + time_steps)])
            ys.append(y[i + time_steps])
        return np.array(Xs), np.array(ys)

    time_steps = 60
    X_seq, y_seq = create_sequences(X, y, time_steps)
    X_seq_t = torch.tensor(X_seq, dtype=torch.float32)
    y_seq_t = torch.tensor(y_seq, dtype=torch.float32)

except Exception as e:
    st.error(f"Data loading error: {e}")
    st.stop()

# ---------------------------
# 4. Define model
# ---------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# ---------------------------
# 5. Load trained model
# ---------------------------
device = torch.device("cpu")
model = LSTMModel(input_dim=len(features), hidden_dim=64, num_layers=2, output_dim=1).to(device)

try:
    model.load_state_dict(torch.load("best_lstm_model.pth", map_location=device))
    model.eval()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# ---------------------------
# 6. Make predictions
# ---------------------------
with torch.no_grad():
    y_pred_scaled = model(X_seq_t).cpu().numpy()

y_true = scaler_y.inverse_transform(y_seq)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

rmse = sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# ---------------------------
# 7. Display results
# ---------------------------
st.subheader("📊 Model Performance")
st.write(f"**RMSE:** {rmse:.3f}")
st.write(f"**MAE:** {mae:.3f}")
st.write(f"**MAPE:** {mape:.2f}%")

st.subheader("📈 Recent Predictions")
pred_df = pd.DataFrame({
    "Date": df.index[-len(y_pred):],
    "Actual": y_true.flatten(),
    "Predicted": y_pred.flatten()
})
st.line_chart(pred_df.set_index("Date")[["Actual", "Predicted"]])

# ---------------------------
# 8. Next-day prediction
# ---------------------------
last_sequence = torch.tensor(X[-time_steps:], dtype=torch.float32).unsqueeze(0).to(device)
next_day_pred_scaled = model(last_sequence).cpu().detach().numpy()
next_day_pred = scaler_y.inverse_transform(next_day_pred_scaled)[0][0]

st.subheader("📅 Next-Day Forecast")
st.write(f"**Predicted next-day closing price:** ${next_day_pred:.2f}")

st.success("✅ Prediction complete!")

