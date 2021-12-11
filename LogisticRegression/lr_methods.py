import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class MAP:
    
    def __init__(self, std):

        self.std = std

    def J(self, x, y, w):

        return np.log(1 + np.exp(-y*np.dot(w,x))) + (np.dot(w,w)/(self.std**2))

    def dJdW(self, x, y, w, M):

        return (M * (sigmoid(y*np.dot(w,x)) - 1) * (y*x)) + ((2/self.std)*w)

class MLE:
    
    def __init__(self, _):
        pass

    def J(self, x, y, w):

        return np.log(1 + np.exp(-y*np.dot(w,x)))

    def dJdW(self, x, y, w, M):
       
        return M * (sigmoid(y*np.dot(w,x)) - 1) * (y*x)
