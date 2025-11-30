import pandas as pd
import numpy as np
import math
import mlflow
import mlflow.sklearn

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ==== Load dataset ====
df = pd.read_csv("Membangun_model/aapl.us.txt_preprocessing.csv")

# Menggunakan fitur Close_norm
dataset = df["Close_norm"].values.reshape(-1, 1)

# ==== Windowing function ====
def create_window(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# ==== Split Train/Test ====
training_data_len = math.ceil(len(dataset) * 0.8)

train_data = dataset[:training_data_len]
test_data  = dataset[training_data_len - 60:]

X_train, y_train = create_window(train_data, 60)
X_test,  y_test  = create_window(test_data, 60)

# ==== MLflow setup ====
mlflow.set_tracking_uri("mlruns")  # lokal, tidak perlu server
mlflow.set_experiment("stock_prediction")

with mlflow.start_run():
    mlflow.autolog()

    model = SVR(kernel="rbf", C=100, gamma=0.1)
    model.fit(X_train, y_train)

    # ==== Evaluation ====
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("RMSE:", rmse)

    mlflow.log_metric("RMSE", rmse)

    mlflow.sklearn.log_model(model, "model")
