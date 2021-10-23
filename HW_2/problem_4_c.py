import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
sys.path.append('..\..')

from MLPy.LinearRegression.sgd import SGD

train = np.load('concrete/train.npy')
train = np.load('concrete/test.npy')

x, y = np.hstack((np.ones((train.shape[0], 1)),
    train[:,:-1])), train[:,-1].reshape(train[:,-1].shape[0], 1)
x = x.T
y = y.T

a = np.matmul(x, x.T)
import pdb;pdb.set_trace()
b = np.matmul(x, y.T)
w_star = np.matmul(np.linalg.inv(a), b)
import pdb;pdb.set_trace()

print(w_star)
