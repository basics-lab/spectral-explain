from sklearn.neural_network import MLPRegressor
import numpy as np

class NeuralNetwork:
    def __init__(self, signal, b):
        n = signal.n
        coordinates = []
        values = []
        for m in range(len(signal.all_samples)):
            for d in range(len(signal.all_samples[0])):
                for z in range(2 ** b):
                    coordinates.append(signal.all_queries[m][d][z])
                    values.append(np.real(signal.all_samples[m][d][z]))
        coordinates = np.real(np.array(coordinates))
        values = np.real(np.array(values))
        self.nn = MLPRegressor(hidden_layer_sizes=(n // 2, n // 4, n // 8),
                            max_iter=500, random_state=0).fit(coordinates, values)

    def evaluate(self, saved_samples):
        query_indices, y_true = saved_samples
        y_hat = self.nn.predict(query_indices)
        return 1 - (np.linalg.norm(y_true - y_hat) ** 2 / np.linalg.norm(y_true - np.mean(y_true)) ** 2)