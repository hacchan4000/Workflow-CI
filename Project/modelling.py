import argparse
import pandas as pd
import numpy as np
import math

import mlflow
import mlflow.keras
from sklearn.metrics import mean_squared_error

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM

# ====== Parsing argument dari MLflow Project ======
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="aapl.us.txt_preprocessing.csv")
args = parser.parse_args()

# ==== Load dataset ====
df = pd.read_csv(args.data_path)
dataset = df["Close_norm"].values.reshape(-1, 1)

# ==== Windowing ====
def create_window(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

training_data_len = math.ceil(len(dataset) * 0.8)

train_data = dataset[:training_data_len]
test_data  = dataset[training_data_len - 60:]

X_train, y_train = create_window(train_data)
X_test,  y_test  = create_window(test_data)

# === Reshape 3D for LSTM ===
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# ==== MLflow Tracking (local runs only, CI later uploads artifacts) ====
mlflow.autolog()

with mlflow.start_run():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")

    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("RMSE:", rmse)
    mlflow.log_metric("RMSE", float(rmse))

    mlflow.keras.log_model(model, "model")
