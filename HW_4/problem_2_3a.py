import numpy as np
import matplotlib.pyplot as plt
import sys
import copy

sys.path.append('../../')
sys.path.append('..\..')

from MLPy.SVM.svm import DualSVM

C = [100/873, 500/873, 700/873]

train = np.genfromtxt('bank-note/train.csv', delimiter=',')
test = np.genfromtxt('bank-note/test.csv', delimiter=',')

objs = []

for c in C:

    pSVM = DualSVM(train,
                     test,
                     c)

    pSVM()
    objs.append(copy.deepcopy(pSVM))
    
    print(f'w_vec: {pSVM.w}')
    print(f'bias: {pSVM.bias}')
    print(f'train error: {pSVM.train_error}')
    print(f'test error: {pSVM.test_error}')

