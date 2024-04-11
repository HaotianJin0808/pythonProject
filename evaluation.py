from sklearn.metrics import accuracy_score, f1_score
from data_preprocessing import load_data_train,load_data_test, split_data
import numpy as np


def evaluate_model(model, X_test, y_test):
    print("X_test shape:", X_test.shape)  # 打印 X_test 的形状来检查
    predictions = model.predict(X_test)
    # 根据自编码器的重构误差确定异常
    threshold = 0.5
    y_pred = [1 if np.mean((p - t) ** 2) > threshold else 0 for p, t in zip(predictions, X_test)]

    accuracy = accuracy_score(y_test, y_pred)
    # 使用 'weighted' 作为多分类问题的平均计算方法
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Model Accuracy: {:.2f}".format(accuracy))
    print("Model F1 Score: {:.2f}".format(f1))

