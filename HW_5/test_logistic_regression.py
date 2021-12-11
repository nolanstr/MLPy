import numpy as np
import sys
sys.path.append('../../')

from MLPy.LogisticRegression.logistic_regression import LogisticRegression

x = np.array([[1, 0.5, -1, 0.3],
              [1, -1, -2, -2],
              [1, 1.5, 0.2, -2.5]])

y = np.array([1,-1,1])

std = 1.0

learn_rates = [0.01, 0.005, 0.0025]

gamma = lambda t: learn_rates[t]

LR = LogisticRegression(x, y, gamma, std, 'MAP')

LR(3)

print(LR.get_test_train_error(x,y))

import pdb;pdb.set_trace()
