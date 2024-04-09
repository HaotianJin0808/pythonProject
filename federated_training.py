from model import create_autoencoder
from data_preprocessing import load_data_train,load_data_test, split_data
import numpy as np

def train_local_models(X_train, y_train, num_clients=3):
    models = []
    chunk_size = len(X_train) // num_clients
    print("X_trian维度：", X_train.shape)
    for i in range(num_clients):
        X_local = X_train[i*chunk_size:(i+1)*chunk_size]
        model = create_autoencoder(X_local.shape[1])
        model.fit(X_local, X_local, epochs=10, batch_size=256, shuffle=True, verbose=0)
        models.append(model)

    for model in models:
        for layer_weights in model.get_weights():
            print(layer_weights.shape)

    return models

# def aggregate_models(models):
#     global_model = create_autoencoder(models[0].input_shape[1])
#     global_weights = np.mean([model.get_weights() for model in models], axis=0)
#     global_model.set_weights(global_weights)
#     return global_model
def aggregate_models(models):
    global_model = create_autoencoder(models[0].input_shape[1])

    # 初始化全局模型的权重
    global_weights = [np.zeros_like(w) for w in models[0].get_weights()]

    # 计算所有模型的权重总和
    for model in models:
        model_weights = model.get_weights()
        for i, w in enumerate(model_weights):
            global_weights[i] += w

    # 计算平均权重
    num_models = len(models)
    global_weights = [w / num_models for w in global_weights]

    # 设置全局模型的权重
    global_model.set_weights(global_weights)

    return global_model
