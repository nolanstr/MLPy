import numpy as np

w = np.array([2, 3])
b = -4

x_vals = np.array([[1, 1],
                   [1, -1],
                   [0, 0],
                   [-1, 3]])

labels = np.array([1,-1,-1,1,1]).reshape((-1,1))


dist = abs(np.matmul(x_vals, w.T) + b) / np.linalg.norm(w)
print(dist)
print('Margin =', dist.min())

import pdb;pdb.set_trace()
