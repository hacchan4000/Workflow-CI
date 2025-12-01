import pandas as pd
import numpy as np
import math
import mlflow
import mlflow.sklearn
import os

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

df = pd.read_csv("aapl.us.txt_preprocessing.csv")
dataset = df["Close_norm"].values.reshape(-1, 1)

def create_window(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

training_data_len = math.ceil(len(dataset) * 0.8)
train_data = dataset[:training_data_len]
test_data  = dataset[training_data_len - 60:]

X_train, y_train = create_window(train_data, 60)
X_test,  y_test  = create_window(test_data, 60)

# Tracking URI
mlflow.set_tracking_uri(f"file://{os.getcwd()}/mlruns")
mlflow.set_experiment("ci_retrain_model")

# FIX: MLflow Project already creates a run, so don't create another
import sys

run_id = sys.argv[1]
mlflow.start_run(run_id=run_id)


model = SVR(kernel="rbf", C=100, gamma=0.1)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

mlflow.log_metric("RMSE", rmse)

input_example = X_test[:1]
signature = mlflow.models.signature.infer_signature(X_train, model.predict(X_train))

mlflow.sklearn.log_model(
    model,
    "model",
    input_example=input_example,
    signature=signature
)
