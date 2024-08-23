import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller

# Load financial and stock data
merged_financial_data = pd.read_csv('Merged_Financial_Data.csv')
stock_data = pd.read_csv('TSLA_Quarterly_Data.csv')

# Ensure column names are consistent
merged_financial_data.columns = merged_financial_data.columns.astype(str)
stock_data.columns = stock_data.columns.astype(str)

# Convert 'Date' columns to datetime format
merged_financial_data['Date'] = pd.to_datetime(merged_financial_data['Date'])
stock_data['Date'] = pd.to_datetime(stock_data['Date'])

# Merge datasets on 'Date'
merged_df = pd.merge(merged_financial_data, stock_data, on='Date', how='inner')

# Prepare features and target
features = merged_df.drop(columns=['Date', 'Close_y'])
target = merged_df['Close_y']

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)

# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Function to test stationarity and return p-value
def test_stationarity(timeseries):
    result = adfuller(timeseries)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print(f'Critical Values {key}: {value}')
    return result[1]  # Return the p-value

# Check if the time series is stationary
p_value = test_stationarity(y_train)

# Use auto_arima to find the best parameters for ARIMA
arima_model = auto_arima(y_train, seasonal=False, trace=True)
arima_order = arima_model.order
arima_model = ARIMA(y_train, order=arima_order).fit()

# Forecast using ARIMA
arima_forecast = arima_model.predict(start=len(y_train), end=len(y_train)+len(y_test)-1, dynamic=False)

# Evaluate ARIMA model
arima_mse = mean_squared_error(y_test, arima_forecast)
arima_mae = mean_absolute_error(y_test, arima_forecast)
arima_r2 = r2_score(y_test, arima_forecast)

from sklearn.model_selection import TimeSeriesSplit

# TimeSeriesSplit allows you to split time series data with respect to time order
tscv = TimeSeriesSplit(n_splits=5)

# Ensure y_train is a numpy array to allow indexing
y_train_array = y_train.to_numpy()

for train_index, test_index in tscv.split(X_train):
    X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
    y_train_cv, y_test_cv = y_train_array[train_index], y_train_array[test_index]

# XGBoost model training with time series split
param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_imputed, y_train)
best_xgb_model = grid_search.best_estimator_

# Make predictions with XGBoost
y_test_pred_xgb = best_xgb_model.predict(X_test_imputed)

# Evaluate XGBoost model
xgb_mse = mean_squared_error(y_test, y_test_pred_xgb)
xgb_mae = mean_absolute_error(y_test, y_test_pred_xgb)
xgb_r2 = r2_score(y_test, y_test_pred_xgb)

# Transformer Model definition and training
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.2):
        super(TimeSeriesTransformer, self).__init__()
        self.model_dim = model_dim
        self.input_embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        x = self.input_embedding(x)
        x += self.positional_encoding
        x = self.transformer_encoder(x)
        return self.fc_out(x[:, -1, :])

input_dim = X_train_imputed.shape[1]
model_dim = 128
num_heads = 8
num_layers = 3
output_dim = model_dim
dropout = 0.3

transformer_model = TimeSeriesTransformer(input_dim, model_dim, num_heads, num_layers, output_dim, dropout)
criterion = nn.MSELoss()
optimizer = optim.Adam(transformer_model.parameters(), lr=0.0005)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_imputed, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Training loop for the improved Transformer
num_epochs = 100
for epoch in range(num_epochs):
    transformer_model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = transformer_model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

# Extract features from the improved Transformer model
transformer_model.eval()
with torch.no_grad():
    train_features = transformer_model(X_train_tensor).numpy()
    test_features = transformer_model(torch.tensor(X_test_imputed, dtype=torch.float32).unsqueeze(1)).numpy()

# XGBoost Model Training with Grid Search for Hyperparameter Tuning
grid_search.fit(train_features, y_train)
best_xgb_model = grid_search.best_estimator_

# Make predictions with the best XGBoost model
y_test_pred_xgb_transformer = best_xgb_model.predict(test_features)

# Evaluate the improved Hybrid model
xgb_transformer_mse = mean_squared_error(y_test, y_test_pred_xgb_transformer)
xgb_transformer_mae = mean_absolute_error(y_test, y_test_pred_xgb_transformer)
xgb_transformer_r2 = r2_score(y_test, y_test_pred_xgb_transformer)

# Final Evaluation Summary
print("Final Model Evaluation Summary:")
print(f"ARIMA Model -> MSE: {arima_mse:.4f}, MAE: {arima_mae:.4f}, R2 Score: {arima_r2:.4f}")
print(f"XGBoost Model -> MSE: {xgb_mse:.4f}, MAE: {xgb_mae:.4f}, R2 Score: {xgb_r2:.4f}")
print(f"Hybrid Transformer + XGBoost Model -> MSE: {xgb_transformer_mse:.4f}, MAE: {xgb_transformer_mae:.4f}, R2 Score: {xgb_transformer_r2:.4f}")

# Optional: Save the best model for future use
import joblib
joblib.dump(best_xgb_model, 'best_xgboost_model.pkl')

# Optional: Save the trained Transformer model
torch.save(transformer_model.state_dict(), 'transformer_model.pth')

print("Models saved successfully.")

