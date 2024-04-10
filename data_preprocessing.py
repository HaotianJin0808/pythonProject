import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import numpy as np

def load_data_train(csv_file):
    df = pd.read_csv(csv_file)
    #删除无用列
    columns_to_delete = ['Flow ID', 'Src IP', 'Src Port','Dst IP','Dst Port','Timestamp','Flow Duration']
    df = df.drop(columns_to_delete, axis=1)

    # 假设最后一列是标签，其余为特征
    label_mapping = {'NORMAL': 0, 'c_ci_na_1': 1}
    df['Label'] = df['Label'].map(label_mapping)
    df.to_csv('c_ci_na_1_train.csv', index=False)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # 将无限大值替换为NaN
    X = np.where(np.isinf(X), np.nan, X)

    # 将DataFrame中的NaN值替换或删除
    X_df = pd.DataFrame(X)
    X_df.fillna(X_df.mean(), inplace=True)  # 用每列的平均值填充NaN值
    # 或者，如果你更愿意删除这些行，可以使用 X_df.dropna(inplace=True)

    # 将处理后的DataFrame转换回NumPy数组
    X_cleaned = X_df.values

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cleaned)
    print(y.shape)

    return X_scaled, y

def load_data_test(csv_file):
    df = pd.read_csv(csv_file)
    #删除无用列
    columns_to_delete = ['Flow ID', 'Src IP', 'Src Port','Dst IP','Dst Port','Timestamp','Flow Duration']
    df = df.drop(columns_to_delete, axis=1)

    # 假设最后一列是标签，其余为特征
    label_mapping = {'NORMAL': 0, 'c_ci_na_1': 1}
    df['Label'] = df['Label'].map(label_mapping)
    df.to_csv('c_ci_na_1_test.csv', index=False)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # 将无限大值替换为NaN
    X = np.where(np.isinf(X), np.nan, X)

    # 将DataFrame中的NaN值替换或删除
    X_df = pd.DataFrame(X)
    X_df.fillna(X_df.mean(), inplace=True)  # 用每列的平均值填充NaN值
    # 或者，如果你更愿意删除这些行，可以使用 X_df.dropna(inplace=True)

    # 将处理后的DataFrame转换回NumPy数组
    X_cleaned = X_df.values

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cleaned)
    print(y.shape)

    return X_scaled, y


def split_data(X, y, test_size=0.2):
    # 将数据分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)


    return X_train, X_test, y_train, y_test
