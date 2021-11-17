import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
sys.path.append('..\..')

from MLPy.SVM.svm import DualSVM

y_0 = 0.5
a = 0.01
gamma_fnc = lambda i: y_0/ (1 + (y_0*i/a))

C = [100/873, 500/873, 700/873]

train = np.genfromtxt('bank-note/train.csv', delimiter=',')
test = np.genfromtxt('bank-note/test.csv', delimiter=',')

for c in C:

    pSVM = DualSVM(train,
                     test,
                     c,
                     gamma_fnc)

    pSVM()

