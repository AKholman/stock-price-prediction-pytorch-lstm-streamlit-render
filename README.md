# ML-Powered Pipeline for Daily AAPL Stock Price Forecasting

Goal: Predict next-day close price for AAPL (daily regression);
Dataset: Daily OHLCV+Adj_Close from Yahoo Finance, last 5 years (~1250 rows);
Pipeline: data download → preprocessing → train/test split → feature engineering → model → deployment;
Moetrics: MAE/RMSE on latest predictions;

Models trained and tested: 
Sarima (classical TSA);
RandomRorest; 
XGBoost;
LSTM (Deep Learning approach, using Tensorflow/keras, Pytorch)


LSTM model with Pytorch shows the best performance. Therefore, below we describe some details of this model traning and testing. 

1.  Using Min–Max scaling we scaled (normalized) the features (X) and the target (y). The scalers are fit only on the training data, then applied to validation and test.

2. Sequence creation - 'def create_sequences(X, y, time_steps=60)' : 
Goal of this step - Transform each continuous 1D timeline of features into overlapping time windows (sequences).
Each sequence of time_steps = 60 days becomes one sample for the LSTM, and the label is the target value right after that window. 
Output: we have NumPy arrays (X_train_seq, y_train_seq, etc.).

But PyTorch models can only work with PyTorch tensors with GPU acceleration and automatic differentiation (autograd).

3. So, we converted all the NumPy arrays (matrix) into PyTorch tensors and prepares them for mini-batch training.
X(rows, features) -> X(rows, timestep, features), i.e. 2D data (matrix) → 3D sequences (tensor)	LSTM needs (batch, time, features). 
DataLoader (a PyTorch utility) breaks the dataset into mini-batches of 32 samples.

4. LSTM model:
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # take the last time step
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

5. One full batch cycle of training: 
    🔹 Step 1 — Forward pass:
        Input batch → LSTM1 → LSTM2 → fc1 → ReLU → fc2 → Output → Prediction.

    🔹 Step 2 — Compute loss (prediction vs actual value):
        Loss = criterion(output, true_y). (e.g., - Mean Squared Error for regression)

    🔹 Step 3 — Backpropagation: 
        Backward pass & weight update (training phase) — this is where LSTM learns
            Call loss.backward(): 
                → loss.backward() - PyTorch automatically computes gradients ∂loss/∂weight for all layers.

        During BackPropagation: 
            PyTorch uses autograd to trace all operations during forward pass.
            It then applies BackPropagation Through Time (BPTT) to move gradients backward through:

                fc2 → fc1 → LSTM2 → LSTM1

            And through time across all 60 timesteps inside each LSTM layer (the “through time” part).
            So, every weight that influenced the prediction gets its gradient value calculated.

    🔹 Step 4 — Optimizer 
        Optimizer - torch.optim.Adam(model.parameters(), lr=0.001) - definition of optimizer;
            lr=0.001 - learning rate;
            
            optimizer.zero_grad() - before going to the next batch, clear old gradients;
            
            optimizer.step() - starts new batch with updates all weights and biases in:
                LSTM1 (input weights, recurrent weights, biases)
                LSTM2 (same)
                fc1 (weights + bias)
                fc2 (weights + bias)
    
🔹 Step 5  — Repeat for every batch
        Each batch computes its own loss and gradients.
        Weights are updated after each batch (this is what makes it stochastic gradient descent).
        After all batches are processed → one epoch is done.
        After many epochs, model gradually converges.

        Thus, each batch changes the model slightly — so the next batch starts with slightly improved weights.


🔹 Step 6: Model Evaluation & Inference - final evaluation and prediction phase.
    PyTorch handles weight storage and reloading into the right layers automatically.
    with torch.no_grad():
        y_pred_scaled = model(X_test_t).cpu().numpy()   - this code disables the gradient computation.
        Reverse Scaling (Back to Original Prices)
    Evaluating Performance using RMSE and MAE. 



<p align="center">
  <img src="https://github.com/AKholman/stock-price-prediction-pytorch-lstm-streamlit-render/blob/main/Graph_2.png?raw=true" width="600"/>
  <br>
</p>

NN layers: 

<p align="center">
  <img src="https://github.com/AKholman/stock-price-prediction-pytorch-lstm-streamlit-render/blob/main/Graph_4.png?raw=true" width="600"/>
  <br>
</p>


<p align="center">
  <img src="https://github.com/AKholman/stock-price-prediction-pytorch-lstm-streamlit-render/blob/main/Graph.png?raw=true" width="600"/>
  <br>
</p>










Quick checklist you can copy into a project README

 Project Spec + success metrics (business & ML)
 Data pipeline: yfinance downloader + raw snapshots. GitHub
 ETL DAG (Airflow/Prefect) + data validation. Apache Airflow
 Feature store + train/test split by time.
 Experiments tracked to MLflow + model registry. MLflow
 Backtests with slippage & fees (vectorbt/backtesting.py). VectorBT +1
 Containerized serving + monitoring dashboard.
 Retraining & rollback procedure.


The project covers the following part of MLSD:

✅ Clarify – Define task (forecast next-day AAPL price).
✅ Metrics & SLAs – RMSE/MAE defined (but no SLA guarantees).
✅ Data – Yahoo Finance → preprocessing pipeline.
✅ Model choice & features – Choose TSA model(s).
✅ Training & infra – Train model locally/in-app.
✅ Serving & scaling – Deploy on Streamlit + Render.
✅ Monitoring & tradeoffs – Basic (manual monitoring, dynamic re-fetch ensures live data).

⚠️ Not included / partially:
Automated pipelines (Airflow, Prefect).
Continuous monitoring (Prometheus, Grafana).
Advanced CI/CD + scaling infra (Kubernetes, SageMaker).
Business SLAs (availability, latency guarantees).


# ==========================
# Project: AAPL Next-Day Close Price Prediction
# Type: Daily Regression
# Full MLSD Pipeline
# ==========================

# 1. Goal Definition
# ------------------
goal = "Predict next-day close price for AAPL using daily OHLCV data"
task = Regression
# 2. Metrics & SLAs
# -----------------
metrics = ["MAE", "RMSE", "R2"]
sla = {"prediction_latency": "< 1 second per request",
    "accuracy_threshold": "MAE < 3 USD"}

# 3. Data Collection
# ------------------
data_source = "Yahoo Finance"
ticker = "AAPL"
frequency = "Daily"
lookback_period = "Last 5 years (~1250 rows)"

# 4. Train/Test Split
# ------------------
train_test_split = {"train": "70-80% of historical data",
                     "test": "20-30% of historical data"}
# ------------------
# 5. Model Choice & Training
# --------------------------
models = ["RandomForestRegressor", "XGBoostRegressor", "LinearRegression"]
training_steps = [
    "Initialize model",
    "Fit model on training data",
    "Hyperparameter tuning (GridSearchCV / RandomSearch)",
    "Validate on test set",
    "Select best-performing model"
]

# 6. Deployment / Serving
# -----------------------
deployment = {
    "framework": "Streamlit / Flask / FastAPI",
    "server": "Render / Heroku / AWS",
    "endpoint": "/predict_next_day_close"
}

# 7. Monitoring & Evaluation
# --------------------------
monitoring = [
    "Track performance metrics (MAE, RMSE) on new predictions",
    "Detect concept drift in stock prices",
    "Log predictions and input features for auditing"
]

# 8. Retraining & Feedback Loop
# -----------------------------
retraining_policy = {
    "frequency": "Weekly / Monthly",
    "trigger": "If MAE exceeds threshold or drift detected",
    "process": "Re-run training pipeline with updated historical data"
}

# 9. Tradeoffs & Considerations
# -----------------------------
tradeoffs = [
    "Latency vs accuracy",
    "Single-stock vs multi-stock scalability",
    "Feature complexity vs interpretability"
]

# End of MLSD Pipeline


**APPENDICES** 

A) LSTM MODEL BUILDING:

Input shape: (60 timesteps, 6 features)
      ▼
┌─────────────────────────────┐
│ LSTM(64, return_sequences=True) │
│ → outputs 64 features per timestep │
│ Output shape: (60, 64)            │
└─────────────────────────────┘
      ▼
 Dropout(0.2)
      ▼
┌─────────────────────────────┐
│ LSTM(32, return_sequences=False) │
│ → outputs only final timestep vector │
│ Output shape: (32,)               │
└─────────────────────────────┘
      ▼
 Dropout(0.2)
      ▼
┌─────────────────────────────┐
│ Dense(16, activation='relu') │
│ → fully connected layer, learns nonlinear combinations │
│ Output shape: (16,)          │
└─────────────────────────────┘
      ▼
┌─────────────────────────────┐
│ Dense(1) │
│ → final prediction (regression) │
│ Output shape: (1,)             │
└─────────────────────────────┘


Total layers: 4
Input layer: (60, 6) → 6 (features) × 60 (timestep)
First LSTM layer: 64 neurons (each learning a temporal pattern)
Output layer: 1 neuron (final continuous prediction)