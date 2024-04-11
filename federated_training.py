from model import create_autoencoder
from keras.callbacks import Callback
from sklearn.metrics import accuracy_score, f1_score
from data_preprocessing import load_data_train,load_data_test, split_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


class EvaluationMetrics(Callback):
    def __init__(self, X_test, y_test, threshold=0.5):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.threshold = threshold
        self.accuracies = []
        self.f1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.X_test)
        y_pred = [1 if np.mean((p - t) ** 2) > self.threshold else 0 for p, t in zip(predictions, self.X_test)]

        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='binary')

        self.accuracies.append(accuracy)
        self.f1_scores.append(f1)

        print(f"Epoch: {epoch + 1}, Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}")
# 最后的版本：报文识别
# def train_local_models(X_train, y_train, X_test,y_test,num_clients=3):
#     models = []
#     histories = []  # 用于记录每个模型的训练历史
#     chunk_size = len(X_train) // num_clients
#     print("X_trian维度：", X_train.shape)
#
#     evaluation_metrics = EvaluationMetrics(X_test, y_test)
#
#     for i in range(num_clients):
#         X_local = X_train[i*chunk_size:(i+1)*chunk_size]
#         model = create_autoencoder(X_local.shape[1])
#
#         history=model.fit(X_local, X_local, epochs=10, batch_size=256, shuffle=True, verbose=0,callbacks=[evaluation_metrics])
#         models.append(model)
#         histories.append(history.history)  # 记录每个模型的训练历史
#         # 绘制损失函数随着迭代次数的变化图像
#         plt.plot(history.history['loss'], label='client {}'.format(i))
#     # for model in models:
#     #     for layer_weights in model.get_weights():
#     #         print(layer_weights.shape)
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.show()
#
#     # 绘制准确率和F1值随迭代次数变化的图像
#     plt.figure(figsize=(12, 5))
#
#     plt.subplot(1, 2, 1)
#     plt.plot(evaluation_metrics.accuracies, label='Accuracy')
#     plt.title('Accuracy vs. Epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend()
#
#     plt.subplot(1, 2, 2)
#     plt.plot(evaluation_metrics.f1_scores, label='F1 Score')
#     plt.title('F1 Score vs. Epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('F1 Score')
#     plt.legend()
#
#     plt.tight_layout()
#     plt.show()
#
#     return models,histories

# def train_local_models(client_datasets, X_test, y_test, epochs=10, batch_size=256):
#     models = []
#     histories = []  # 用于记录每个模型的训练历史
#
#     evaluation_metrics = EvaluationMetrics(X_test, y_test)
#
#     for i, dataset in enumerate(client_datasets):
#         X_local, y_local = dataset
#         print("X_local shape:", X_local.shape)  # 这里添加打印语句来检查 X_local 的形状
#         model = create_autoencoder(X_local.shape[1])
#
#         history = model.fit(X_local, X_local, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0, callbacks=[evaluation_metrics])
#
#         models.append(model)
#         histories.append(history.history)  # 记录每个模型的训练历史
#
#     return models, histories
def train_local_models(client_datasets, X_test, y_test, epochs=18, batch_size=64):
    models = []
    histories = []  # 用于记录每个模型的训练历史

    evaluation_metrics = EvaluationMetrics(X_test, y_test)

    for i, dataset in enumerate(client_datasets):
        X_local, y_local = dataset
        print(f"Client {i+1}, X_local shape: {X_local.shape}")  # 打印 X_local 的形状

        model = create_autoencoder(X_local.shape[1])
        print(f"Client {i+1}, Model created.X_local shape: {X_local.shape}")  # 确认模型已创建

        history = model.fit(X_local, X_local, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)
        print(f"Client {i+1}, Model trained.")  # 确认模型已训练

        models.append(model)
        histories.append(history.history)  # 记录每个模型的训练历史

    print("All models trained successfully.")
    return models, histories

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
