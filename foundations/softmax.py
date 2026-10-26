import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        z -= np.max(z)
        return np.round(np.exp(z)/np.sum(np.exp(z)),4)
