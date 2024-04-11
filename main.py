from data_preprocessing import load_data_train,load_data_test, split_data
from federated_training import train_local_models, aggregate_models
from evaluation import evaluate_model

# if __name__ == "__main__":
#     # 加载和预处理数据
#     X_train, y_train = load_data_train('D:/python_code/pythonProject/data_list/c_ci_na_1_training_data.csv')
#     X_test, y_test = load_data_test('D:/python_code/pythonProject/data_list/test_data.csv')
#     X_train, X_test, y_train, y_test = split_data(X_train, y_train, test_size=0.2)
#
#     # 训练本地模型
#     local_models,histories = train_local_models(X_train, y_train,X_test,y_test)
#
#     # 聚合模型
#     global_model = aggregate_models(local_models)
#
#     # 评估模型
#     evaluate_model(global_model, X_test, y_test)

if __name__ == "__main__":
    # 加载每个客户端特定的数据集
    client1_data = load_data_train('c_ci_na_1_training_data.csv')
    X_train1, y_train1 = load_data_train('c_ci_na_1_training_data.csv')
    print("X_train1 shape:", X_train1.shape)

    client2_data = load_data_train('c_sc_na_1_training_data.csv')
    X_train2, y_train2 = load_data_train('c_sc_na_1_training_data.csv')
    print("X_train2 shape:", X_train2.shape)

    client3_data = load_data_train('c_se_na_1_training_data.csv')
    X_train3, y_train3 = load_data_train('c_se_na_1_training_data.csv')
    print("X_train3 shape:", X_train3.shape)

    # 加载全局测试集
    X_test, y_test = load_data_test('test_data.csv')

    # 训练本地模型
    client_datasets = [client1_data, client2_data, client3_data]
    local_models, histories = train_local_models(client_datasets, X_test, y_test)

    # 聚合模型
    global_model = aggregate_models(local_models)

    # 评估全局模型
    evaluate_model(global_model, X_test, y_test)

