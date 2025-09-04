# ==========================
# Stock Price Prediction using LSTM and GRU
# Author: Haseeb
# Purpose: Compare LSTM and GRU performance on predicting stock closing prices
# ==========================

# --------------------------
# 1. Import Libraries
# --------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam

# --------------------------
# 2. Load Dataset
# --------------------------
df = pd.read_csv('/content/drive/MyDrive/stock_price_dataset.csv',
                 parse_dates=['Date'], index_col='Date')

# Remove duplicates and handle missing values
df.drop_duplicates(inplace=True)
df = df.interpolate(method='linear')  # Fill missing values
df = df.dropna()

# --------------------------
# 3. Visualize Data
# --------------------------
categorical_features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

for name in categorical_features:
    # Histogram
    sns.histplot(df[name], kde=True)
    plt.xlabel(name)
    plt.ylabel('Frequency')
    plt.title(f"Histogram: {name}")
    plt.show()

    # Box plot
    sns.boxplot(x=df[name])
    plt.xlabel(name)
    plt.ylabel('Value')
    plt.title(f"Box Plot: {name}")
    plt.show()

# --------------------------
# 4. Outlier Removal Function
# --------------------------
def remove_outliers(df, column):
    """
    Remove outliers using Interquartile Range (IQR) method.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# Apply outlier removal
for name in categorical_features:
    df = remove_outliers(df, name)
    # Optional: visualize after outlier removal
    sns.histplot(df[name], kde=True)
    plt.xlabel(name)
    plt.ylabel('Frequency')
    plt.title(f"Histogram after Outlier Removal: {name}")
    plt.show()

    sns.boxplot(x=df[name])
    plt.xlabel(name)
    plt.ylabel('Value')
    plt.title(f"Box Plot after Outlier Removal: {name}")
    plt.show()

# --------------------------
# 5. Select Main Column
# --------------------------
# We only predict 'Close' price
df = df[['Close']]

# --------------------------
# 6. Scale Data
# --------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)

# --------------------------
# 7. Create Time-Series Sequences
# --------------------------
def create_sequence(df_scaled, time_step=60):
    """
    Create sequences of data for time series prediction.
    X = previous time_step days
    y = next day value
    """
    X, y = [], []
    for i in range(len(df_scaled) - time_step - 1):
        X.append(df_scaled[i:(i + time_step), 0])
        y.append(df_scaled[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 100
X, y = create_sequence(df_scaled, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for LSTM/GRU

# --------------------------
# 8. Train-Test Split
# --------------------------
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# --------------------------
# 9. Build LSTM Model
# --------------------------
model_lstm = Sequential()
model_lstm.add(LSTM(128, return_sequences=True, input_shape=(time_step, 1)))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(64, return_sequences=True))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(32))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(1))

# --------------------------
# 10. Build GRU Model
# --------------------------
model_gru = Sequential()
model_gru.add(GRU(128, return_sequences=True, input_shape=(time_step, 1)))
model_gru.add(Dropout(0.2))
model_gru.add(GRU(64))
model_gru.add(Dropout(0.2))
model_gru.add(Dense(1))

# --------------------------
# 11. Compile Models
# --------------------------
optimizer = Adam(learning_rate=0.001)
model_lstm.compile(optimizer=optimizer, loss='mean_squared_error')
model_gru.compile(optimizer=optimizer, loss='mean_squared_error')

# --------------------------
# 12. Train Models
# --------------------------
history_lstm = model_lstm.fit(X_train, y_train,
                              validation_data=(X_test, y_test),
                              epochs=120, batch_size=32, verbose=1)

history_gru = model_gru.fit(X_train, y_train,
                            validation_data=(X_test, y_test),
                            epochs=120, batch_size=32, verbose=1)

# --------------------------
# 13. Predict Next Day Price (Optional)
# --------------------------
input_sequence = df_scaled[-time_step:]
input_sequence = input_sequence.reshape(1, time_step, 1)

predicted_lstm = model_lstm.predict(input_sequence)
predicted_gru = model_gru.predict(input_sequence)

predicted_lstm = scaler.inverse_transform(predicted_lstm)
predicted_gru = scaler.inverse_transform(predicted_gru)

print(f"Predicted Price (LSTM): {predicted_lstm[0][0]:.2f}")
print(f"Predicted Price (GRU): {predicted_gru[0][0]:.2f}")

# --------------------------
# 14. Plot Loss Curves
# --------------------------
plt.figure(figsize=(6, 6))
plt.plot(history_lstm.history['loss'], label='Train LSTM Loss')
plt.plot(history_lstm.history['val_loss'], label='Validation LSTM Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('LSTM Loss Curve')
plt.show()

plt.figure(figsize=(6, 6))
plt.plot(history_gru.history['loss'], label='Train GRU Loss')
plt.plot(history_gru.history['val_loss'], label='Validation GRU Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('GRU Loss Curve')
plt.show()

# --------------------------
# 15. Predict on Test Set for Visualization
# --------------------------
# LSTM Predictions
y_pred_lstm = model_lstm.predict(X_test)
y_pred_lstm = scaler.inverse_transform(y_pred_lstm)
y_test_lstm = scaler.inverse_transform(y_test.reshape(-1, 1))

# GRU Predictions
y_pred_gru = model_gru.predict(X_test)
y_pred_gru = scaler.inverse_transform(y_pred_gru)
y_test_gru = scaler.inverse_transform(y_test.reshape(-1, 1))

# --------------------------
# 16. Plot Actual vs Predicted Prices
# --------------------------
plt.figure(figsize=(12, 6))
plt.plot(y_test_lstm, label='Actual Price')
plt.plot(y_pred_lstm, label='Predicted Price (LSTM)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('LSTM: Actual vs Predicted Prices')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(y_test_gru, label='Actual Price')
plt.plot(y_pred_gru, label='Predicted Price (GRU)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('GRU: Actual vs Predicted Prices')
plt.legend()
plt.show()