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

1.  Using Min–Max scaling we scaled (normalized) the features (X) and the target (y). The scalers are fit only on the training data, then applied to validation and test — which prevents information leakage.

2. Sequence creation - 'def create_sequences(X, y, time_steps=60)' : 
Goal of this step - Transform each continuous 1D timeline of features into overlapping time windows (sequences).
Each sequence of time_steps = 60 days becomes one sample for the LSTM, and the label is the target value right after that window. 
Output: we have NumPy arrays (X_train_seq, y_train_seq, etc.).

But PyTorch models can only work with PyTorch tensors with GPU acceleration and automatic differentiation (autograd).

3. So, Step 4 converts all the NumPy arrays (matrix) into PyTorch tensors and prepares them for efficient mini-batch training.
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
        Input batch → LSTM1 → LSTM2 → fc1 → ReLU → fc2 → Output → Compute prediction.

    🔹 Step 2 — Compute loss:
        Loss = criterion(output, true_y). (e.g., Mean Squared Error for regression)

    🔹 Step 3 — Backpropagation:
        Call loss.backward(): 
            → PyTorch automatically computes gradients ∂loss/∂weight for all layers.

    🔹 Step 4 — Optimizer update:
        optimizer.step()
            → All layer parameters are adjusted based on gradients.

    🔹 Step 5 — Next batch:
        LSTM starts fresh with new sequence inputs.
            Gradients from the previous batch are cleared (optimizer.zero_grad()).
                The updated weights now slightly better fit the data → model improves.



<p align="center">
  <img src="https://github.com/AKholman/stock-price-prediction-pytorch-lstm-streamlit-render/blob/main/Graph_2.png?raw=true" width="600"/>
  <br>
</p>

NN layers: 

<p align="center">
  <img src="https://github.com/AKholman/stock-price-prediction-pytorch-lstm-streamlit-render/blob/main/Graph_3.png?raw=true" width="600"/>
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
sla = {
    "prediction_latency": "< 1 second per request",
    "accuracy_threshold": "MAE < 3 USD"
}

# 3. Data Collection
# ------------------
data_source = "Yahoo Finance"
ticker = "AAPL"
frequency = "Daily"
lookback_period = "Last 5 years (~1250 rows)"
data_pipeline = [
    "Download CSV from Yahoo Finance",
    "Load CSV into pandas DataFrame",
    "Handle missing values",
    "Feature engineering (e.g., moving averages, lag features)"
]

# 4. Train/Test Split
# ------------------
train_test_split = {
    "train": "70-80% of historical data",
    "test": "20-30% of historical data"
}

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


1. MODEL DEFINITION:

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.summary()

🔹 Step 4. Data as it flows:

Step	Layer	      Input shape	     Output shape
1	    Input	       (60, 6)	   →     (60, 6).      usually this is not considered as a layer
2	    LSTM(64)	   (60, 6)	   →     (60, 64)
3	    Dropout(0.2)   (60, 64)    →     (60, 64)
4	    LSTM(32)	   (60, 64)	   →     (32,)
5	    Dropout(0.2)	(32,)	   →     (32,)
6	    Dense(16, ReLU)	(32,)	   →     (16,)
7	    Dense(1)	    (16,)	   →     (1,)

SUMMARY:
Total layers: 7
Input layer: (60, 6) → 6 features × 60 timesteps
First LSTM layer: 64 neurons (each learning a temporal pattern)
Output layer: 1 neuron (final continuous prediction)


2. STEP_BY_STEP DESCRIPTION: 

Sequential([...])

A Keras model type that stacks layers one after another. Easier for simple models (like your LSTM).
Internally, TensorFlow builds a computational graph of operations for forward and backward passes.

LSTM(64, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
LSTM layer → type of RNN for sequence data (time series, text, etc.).

64 → number of hidden units (neurons in the LSTM cell). More units → more capacity to learn patterns.
return_sequences=True → outputs the full sequence for the next layer.
Why? Because the next LSTM layer expects a sequence as input.

input_shape=(timesteps, features) → shape of one input sample:
X_train_seq.shape[1] = 60 timesteps
X_train_seq.shape[2] = 6 features per timestep

Dropout(0.2)
Regularization layer to prevent overfitting. Randomly sets 20% of the input neurons to zero during training. Helps the model not memorize training data.

LSTM(32, return_sequences=False)
Second LSTM layer stacked on the first. 32 → hidden units (fewer than the first layer)
return_sequences=False → outputs only the last timestep.
Why? Because the next layer is Dense, which expects a single vector, not a sequence.

Dropout(0.2)
Another dropout layer after the second LSTM. Helps prevent overfitting again.

Dense(16, activation='relu')
Fully connected (feedforward) layer. 16 → number of neurons

activation='relu' → non-linear activation function:
ReLU(x) = max(0, x)
Introduces non-linearity → essential for neural networks to learn complex patterns

Dense(1)
Output layer. 1 neuron → predicts a single value (e.g., next-day stock price). No activation → linear output (common for regression)

3. MODEL COMPILATION:

model.compile(optimizer='adam', loss='mse').  optimizer='adam. Adam = Adaptive Moment Estimation. Popular gradient descent algorithm with adaptive learning rate
Handles weight updates efficiently

loss='mse'.  Mean Squared Error. Standard loss for regression tasks.
Measures difference between predicted and actual values

Keras functions involved here:
compile() → prepares the model for training (sets optimizer, loss, metrics)
TensorFlow handles the math behind the scenes (forward pass, backward pass, gradient computation).

4. MODEL SUMMARY
model.summary(): 

Keras method that prints:
Each layer’s name, output shape, number of parameters
Total trainable parameters (weights + biases)
Helps to verify model architecture