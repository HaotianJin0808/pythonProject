from model import create_autoencoder
from data_preprocessing import load_data, split_data
import numpy as np

def train_local_models(X_train, y_train, num_clients=3):
    models = []
    chunk_size = len(X_train) // num_clients
    for i in range(num_clients):
        X_local = X_train[i*chunk_size:(i+1)*chunk_size]
        model = create_autoencoder(X_local.shape[1])
        model.fit(X_local, X_local, epochs=10, batch_size=256, shuffle=True, verbose=0)
        models.append(model)
    return models

def aggregate_models(models):
    global_model = create_autoencoder(models[0].input_shape[1])
    global_weights = np.mean([model.get_weights() for model in models], axis=0)
    global_model.set_weights(global_weights)
    return global_model
