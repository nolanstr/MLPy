import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
sys.path.append('..\..')

from MLPy.LinearRegression.batch_gradient import BatchGradient

train = np.load('concrete/train.npy')
test = np.load('concrete/test.npy')

x_train, y_train = np.hstack((np.ones((train.shape[0], 1)),
                                train[:,:-1])), train[:,-1]
x_test, y_test = np.hstack((np.ones((test.shape[0], 1)),
                                test[:,:-1])), test[:,-1]

w = np.zeros((1,x_train.shape[1]))

r = 0.001

bg = BatchGradient(x_train, y_train, w, r)
bg()

test_cost = bg.compute_cost(x_test, y_test)
print('Test cost', test_cost)

plt.plot(np.arange(len(bg.costs)), bg.costs)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.savefig('BatchGradient')

print(bg.w)
