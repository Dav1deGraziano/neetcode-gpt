import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        return np.round(
            np.maximum(
                -np.sum(
                y_true*np.log(np.clip(y_pred,1e-7,None))
                +(1-y_true)*np.log(np.clip(1-y_pred,1e-7,None))
                )/len(y_true),
                0.0),4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        return np.round(
            np.maximum(
                -np.sum(
                y_true.reshape(-1,order='F') *
                np.log(np.clip(y_pred.reshape(-1,order='F'),1e-7,None))
            )/y_true.shape[0]
                ,0.0
            ),4
        )
