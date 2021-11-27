import numpy as np
import matplotlib.pyplot as plt
import sys
import copy

sys.path.append('../../')
sys.path.append('..\..')

from MLPy.SVM.svm import DualSVM

C = [100/873, 500/873, 700/873]
GAMMAS = [0.1, 0.5, 1, 5, 100]

train = np.genfromtxt('bank-note/train.csv', delimiter=',')
test = np.genfromtxt('bank-note/test.csv', delimiter=',')

objs = []

for c in C:
    
    for gamma in GAMMAS:

        pSVM = DualSVM(train,
                         test,
                         c)

        pSVM(gamma=0.1, kernel='Gaussian')

        objs.append(copy.deepcopy(pSVM))
        
        print(f'C: {c}')
        print(f'$\gamma$: {gamma}')
        print(f'w_vec: {pSVM.w}')
        print(f'bias: {pSVM.bias}')
        print(f'train error: {pSVM.train_error}')
        print(f'test error: {pSVM.test_error}')

import pdb;pdb.set_trace()
