import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        
        z = np.matmul(x,w)+b

        if activation == 'sigmoid':
            return np.round(1/(1+np.exp(-z)),5)
        elif activation == 'relu':
            return np.round(np.maximum(0,z),5)
        else:
            raise ValueError("'activation' must be one of 'sigmoid','relu'")
