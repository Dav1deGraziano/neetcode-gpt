import numpy as np
from numpy.typing import NDArray


class Solution:

    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        n = len(initial_weights)
        N = X.shape[0]

        for i in range(num_iterations):
            nabla = np.empty(n)
            y_hat = self.get_model_prediction(X,initial_weights)
            for j in range(n):
                nabla[j] = self.get_derivative(y_hat,Y,N,X,j)
            if np.max(np.abs(self.learning_rate*nabla))<1e-6:
                break
            initial_weights -= self.learning_rate*nabla
        
        return np.round(initial_weights,5)
