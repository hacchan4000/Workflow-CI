import pandas as pd
import numpy as np
import math
import mlflow
import mlflow.sklearn

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

# ==== Split Train/Test ====
training_data_len = math.ceil(len(dataset) * 0.8)
train_data = dataset[:training_data_len]
test_data  = dataset[training_data_len - 60:]

X_train, y_train = create_window(train_data, 60)
X_test,  y_test  = create_window(test_data, 60)



import os
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
mlflow.set_experiment("ci_retrain_model")

mlflow.autolog()
with mlflow.start_run():
    model = SVR(kernel="rbf", C=100, gamma=0.1)

    # 1️⃣ Fit dulu
    model.fit(X_train, y_train)

    # 2️⃣ Baru boleh infer signature
    signature = mlflow.models.signature.infer_signature(
        X_train, 
        model.predict(X_train)
    )

    # 3️⃣ Input example
    input_example = X_test[:1]

    # 4️⃣ Log model
    mlflow.sklearn.log_model(
        model,
        name="model",
        input_example=input_example,
        signature=signature
    )

    # 5️⃣ Metik akurasi
    akurasi = model.score(X_test, y_test)
    mlflow.log_metric("akurasi", akurasi)
    print("berhasil")
