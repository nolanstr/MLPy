import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
sys.path.append('..\..')

from MLPy.LinearRegression.sgd import SGD

x_train = np.array([[1, 1,-1,2],
                    [1, 1,1,3],
                    [1, -1,1,0],
                    [1, 1,2,-4],
                    [1, 3,-1,-1]])
y_train = np.array([[1],[4],[-1],[-2],[0]])


w = np.zeros((1,x_train.shape[1]))

r = 0.1

bg = SGD(x_train, y_train, w, r)
bg()

print(bg.w)
