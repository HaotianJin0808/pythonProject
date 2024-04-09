from data_preprocessing import load_data_train,load_data_test, split_data
from federated_training import train_local_models, aggregate_models
from evaluation import evaluate_model

if __name__ == "__main__":
    # 加载和预处理数据
    X_train, y_train = load_data_train('20200426_UOWM_IEC104_Dataset_c_ci_na_1_attacker1.pcap_Flow.csv')
    X_test, y_test = load_data_test('20200426_UOWM_IEC104_Dataset_c_ci_na_1_attacker1_iec104_only.pcap_Flow.csv')
    X_train, X_test, y_train, y_test = split_data(X_train, y_train, test_size=0.2)

    # 训练本地模型
    local_models = train_local_models(X_train, y_train)

    # 聚合模型
    global_model = aggregate_models(local_models)

    # 评估模型
    evaluate_model(global_model, X_test, y_test)
