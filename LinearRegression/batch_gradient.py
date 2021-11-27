import numpy as np
from .cost_function import J, DJ

class BatchGradient:

    def __init__(self, x, y, w, r):
        
        self.x = x
        self.y = y.reshape((self.x.shape[0], 1))
        self.w = w.reshape((self.x.shape[1], 1))
        self.r = r
        self.costs = [J(self.x, self.y, self.w)] 

    def __call__(self, tol=10e-6):
        
        for _ in range(10000):
            dJ = self.r * DJ(self.x, self.y, self.w)
            self.w -= dJ 
            self.costs.append(J(self.x, self.y, self.w)) 
            print(self.costs[-1])
            if self.costs[-1] < tol:
                break
            
    def compute_cost(self, x, y):

        return J(x, y, self.w)
