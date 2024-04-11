import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# 1. 加载数据
df = pd.read_csv('/data_list/c_ci_na_1_training_data.csv')
columns_to_delete = ['flow id', 'protocol', 'src ip', 'dst ip', 'src port', 'dst port','flow start timestamp']
df = df.drop(columns_to_delete, axis=1)

# 假设最后一列是标签，其余为特征
label_mapping = {'NORMAL': 0, 'c_ci_na_1': 1}
df['Label'] = df['Label'].map(label_mapping)
# 2. 数据预处理
# 假设最后一列是标签
X = df.iloc[:, :-1]  # 特征
y = df.iloc[:, -1]  # 标签

# 处理分类特征（如果有的话）
# 这里假设所有的列都是数值型，如果有分类数据，请先进行编码
# for col in X.columns:
#     if X[col].dtype == 'object':
#         encoder = LabelEncoder()
#         X[col] = encoder.fit_transform(X[col].astype(str))
#X[np.isinf(X)] = np.nan
# 处理缺失值
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
# 替换无限值
X = np.where(np.isinf(X), np.nan, X)

# 处理过大的数值
# 注意：这里需要你根据数据的具体情况来选择合适的阈值或方法
# 例如，你可以使用一个固定的阈值来替换过大的值，或者使用列的最大值等
# max_value = 1e9  # 假设的阈值
# X = np.where(X > max_value, max_value, X)

# 再次使用SimpleImputer处理NaN值（包括之前替换的无限值）
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# 确保没有无限值或NaN值
if np.any(np.isnan(X)) or np.any(np.isinf(X)):
    raise ValueError("数据仍包含NaN或无限值。")
# 3. 分离训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 训练随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 5. 提取特征重要性并选出前十个
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index=df.columns[:-1],
                                   columns=['importance']).sort_values('importance', ascending=False)

top_10_features = feature_importances.head(15)
print(top_10_features)
