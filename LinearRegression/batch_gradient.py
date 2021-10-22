import numpy as np
from cost_function import J, DJ

class BatchGradient:

    def __init__(x, y, w, r):
        
        self.x = x
        self.y = y
        self.w = w
        self.r = r
        self.costs = [J(self.x, self.y, self.w)] 

    def __call__(self, tol=10e-6):
        
        
        while self.costs[-1] > tol:

            self.w -= (self.r * DJ(self.x, self.y, self.w))
            self.costs.append(J(self.x, self.y, self.w)) 
            

    def compute_cost(self, x, y):

        return J(x, y, self.w)
