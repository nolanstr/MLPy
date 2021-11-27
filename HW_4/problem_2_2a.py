import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
sys.path.append('..\..')

from MLPy.SVM.svm import PrimalSVM

y_0 = 1e-3
a = 1e-4
gamma_fnc = lambda i: y_0/ (1 + (y_0*i/a))

C = [100/873, 500/873, 700/873]

train = np.genfromtxt('bank-note/train.csv', delimiter=',')
test = np.genfromtxt('bank-note/test.csv', delimiter=',')

for c in C:

    pSVM = PrimalSVM(train,
                     test,
                     c,
                     gamma_fnc)

    pSVM(100)
    print(f'bias + w_0: {pSVM.w}')
    print(f'train error: {pSVM.train_error[-1]}')
    print(f'test error: {pSVM.test_error[-1]}')



'''
    plt.plot(np.arange(len(pSVM.J_vals)), pSVM.J_vals, label='J(w)')
    plt.title('Objective Function, J(w)')
    plt.legend()
    plt.grid()
    plt.xlabel('$w_{vals}$')
    plt.ylabel('J(w)')
    plt.show()
    plt.clf()

    plt.plot(np.arange(len(pSVM.train_error)), pSVM.train_error, label='Train')
    plt.title('Train Error')
    plt.legend()
    plt.grid()
    plt.xlabel('$w_{vals}$')
    plt.ylabel('MAE')
    plt.show()
    plt.clf()


    plt.plot(np.arange(len(pSVM.test_error)), pSVM.test_error, label='Test')
    plt.title('Test_error')
    plt.legend()
    plt.grid()
    plt.xlabel('$w_{vals}$')
    plt.ylabel('MAE')
    plt.show()
'''

