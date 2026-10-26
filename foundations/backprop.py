import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        y_hat = 1/(1+np.exp(-np.dot(x,w)-b))
        r2 = (y_hat-y_true)*y_hat*(1-y_hat)
        return (np.round(r2*x,5),np.round(r2,5))
