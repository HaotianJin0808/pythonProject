import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data_train(csv_file):
    df = pd.read_csv(csv_file)
    #删除无用列
    columns_to_delete = ['Flow ID', 'Src IP', 'Src Port','Dst IP','Dst Port','Timestamp']
    df = df.drop(columns_to_delete, axis=1)

    # 假设最后一列是标签，其余为特征
    label_mapping = {'NORMAL': 0, 'c_ci_na_1': 1}
    df['Label'] = df['Label'].map(label_mapping)
    df.to_csv('c_ci_na_1_train.csv', index=False)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(y.shape)

    return X_scaled, y

def load_data_test(csv_file):
    df = pd.read_csv(csv_file)
    #删除无用列
    columns_to_delete = ['Flow ID', 'Src IP', 'Src Port','Dst IP','Dst Port','Timestamp']
    df = df.drop(columns_to_delete, axis=1)

    # 假设最后一列是标签，其余为特征
    label_mapping = {'NORMAL': 0, 'c_ci_na_1': 1}
    df['Label'] = df['Label'].map(label_mapping)
    df.to_csv('c_ci_na_1_test.csv', index=False)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(y.shape)

    return X_scaled, y


def split_data(X, y, test_size=0.2):
    # 将数据分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)


    return X_train, X_test, y_train, y_test
