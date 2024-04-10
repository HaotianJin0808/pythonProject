from model import create_autoencoder
from data_preprocessing import load_data_train,load_data_test, split_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def train_local_models(X_train, y_train, num_clients=3):
    models = []
    histories = []  # 用于记录每个模型的训练历史
    chunk_size = len(X_train) // num_clients
    print("X_trian维度：", X_train.shape)
    for i in range(num_clients):
        X_local = X_train[i*chunk_size:(i+1)*chunk_size]
        model = create_autoencoder(X_local.shape[1])

        history=model.fit(X_local, X_local, epochs=10, batch_size=256, shuffle=True, verbose=0)
        models.append(model)
        histories.append(history.history)  # 记录每个模型的训练历史
        # 绘制损失函数随着迭代次数的变化图像
        plt.plot(history.history['loss'], label='client {}'.format(i))
    # for model in models:
    #     for layer_weights in model.get_weights():
    #         print(layer_weights.shape)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return models,histories

# def aggregate_models(models):
#     global_model = create_autoencoder(models[0].input_shape[1])
#     global_weights = np.mean([model.get_weights() for model in models], axis=0)
#     global_model.set_weights(global_weights)
#     return global_model

# def aggregate_models(models):
#     global_model = create_autoencoder(models[0].input_shape[1])
#
#     # 初始化全局模型的权重
#     global_weights = [np.zeros_like(w) for w in models[0].get_weights()]
#
#     # 计算所有模型的权重总和
#     for model in models:
#         model_weights = model.get_weights()
#         for i, w in enumerate(model_weights):
#             global_weights[i] += w
#
#     # 计算平均权重
#     num_models = len(models)
#     global_weights = [w / num_models for w in global_weights]
#
#     # 设置全局模型的权重
#     global_model.set_weights(global_weights)
#
#     return global_model

def aggregate_models(models):
    global_model = create_autoencoder(models[0].input_shape[1])

    for layer_idx in range(len(global_model.layers)):
        layer_weights = []
        layer_biases = []

        for model in models:
            # 获取当前层的权重
            w = model.layers[layer_idx].get_weights()

            # 如果当前层有权重，则继续处理
            if len(w) > 0:
                weights, biases = w
                layer_weights.append(weights)
                layer_biases.append(biases)

        # 如果当前层有可训练的权重和偏置，则计算平均值并设置给全局模型
        if layer_weights and layer_biases:
            avg_weights = np.mean(np.array(layer_weights), axis=0)
            avg_biases = np.mean(np.array(layer_biases), axis=0)
            global_model.layers[layer_idx].set_weights([avg_weights, avg_biases])

    return global_model
