import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(csv_file):
    df = pd.read_csv(csv_file)
    # 假设最后一列是标签，其余为特征
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def split_data(X, y, test_size=0.2):
    # 将数据分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test
