import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Load data
data = pd.read_csv("reliance.csv", parse_dates=['Date'])
data = data.set_index('Date')

# Feature Engineering
data['50-ma'] = data['Close'].rolling(window=50, min_periods=1).mean()
data['100-ma'] = data['Close'].rolling(window=100, min_periods=1).mean()
data['Close_1'] = data['Close'].shift(1)
data['Close_2'] = data['Close'].shift(2)
data['Close_3'] = data['Close'].shift(3)
data['Return'] = data['Close'].pct_change()
data['Rolling_Mean'] = data['Close'].rolling(window=14).mean()
data['Rolling_Std'] = data['Close'].rolling(window=14).std()
data['Bollinger_Upper'] = data['Rolling_Mean'] + (2 * data['Rolling_Std'])
data['Bollinger_Lower'] = data['Rolling_Mean'] - (2 * data['Rolling_Std'])

data = data.dropna()

# Define features and target
features = ['Close_1', 'Close_2', 'Close_3', '50-ma', '100-ma', 'Bollinger_Upper', 'Bollinger_Lower', 'High', 'Low', 'Volume']
target = 'Close'

# Normalize data
scaler = MinMaxScaler()
data[features + [target]] = scaler.fit_transform(data[features + [target]])

data = data.dropna()

# Prepare data for LSTM
sequence_length = 30

def create_sequences(data, features, target, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[features].iloc[i:i+seq_length].values)
        y.append(data[target].iloc[i+seq_length])
    return np.array(X), np.array(y)

split_index = int(len(data) * 0.8)
train_data = data.iloc[:split_index]
test_data = data.iloc[split_index:]

X_train, y_train = create_sequences(train_data, features, target, sequence_length)
X_test, y_test = create_sequences(test_data, features, target, sequence_length)

# Build LSTM model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(100, return_sequences=False),
    Dropout(0.3),
    Dense(50, activation='relu'),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mse')

# Train model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform predictions
y_pred = scaler.inverse_transform(np.concatenate((np.zeros((y_pred.shape[0], len(features))), y_pred), axis=1))[:, -1]
y_test = scaler.inverse_transform(np.concatenate((np.zeros((y_test.shape[0], len(features))), y_test.reshape(-1, 1)), axis=1))[:, -1]

# Plot actual vs predicted
plt.figure(figsize=(10,5))
plt.plot(test_data.index[sequence_length:], y_test, label="Actual Prices", color='blue')
plt.plot(test_data.index[sequence_length:], y_pred, label="Predicted Prices", color='red', linestyle='dashed')
plt.legend()
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Actual vs Predicted Stock Prices")
plt.show()
