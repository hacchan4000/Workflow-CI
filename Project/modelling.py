import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import math
from sklearn.svm import SVR

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
test_data = dataset[training_data_len - 60:]

X_train, y_train = create_window(train_data, 60)
X_test, y_test = create_window(test_data, 60)

mlflow.autolog()

# TIDAK ADA start_run() DI SINI
model = SVR(kernel="rbf", C=100, gamma=0.1)

model.fit(X_train, y_train)

predict_score = model.score(X_test, y_test)
print("done")
