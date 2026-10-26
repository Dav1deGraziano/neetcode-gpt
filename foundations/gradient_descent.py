class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        
        x = init
        tol = 1e-6

        for i in range(iterations):
            prev = x
            x -= 2*learning_rate*x
            if abs(x-prev)<tol:
                break
        
        return round(x,5)
