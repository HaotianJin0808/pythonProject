from sklearn.metrics import accuracy_score, f1_score
from data_preprocessing import load_data_train,load_data_test, split_data
import numpy as np


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    # 这里需要根据自编码器的重构误差来确定异常，你可能需要设置一个阈值
    # 假设阈值为0.5
    threshold = 0.5
    y_pred = [1 if np.mean((p - t) ** 2) > threshold else 0 for p, t in zip(predictions, X_test)]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')

    print("Model Accuracy: {:.2f}".format(accuracy))
    print("Model F1 Score: {:.2f}".format(f1))
