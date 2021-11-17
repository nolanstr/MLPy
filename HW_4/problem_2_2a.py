import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
sys.path.append('..\..')

from MLPy.SVM.svm import PrimalSVM

y_0 = 0.5
a = 0.01
gamma_fnc = lambda i: y_0/ (1 + (y_0*i/a))

c = 100/873

train = np.genfromtxt('bank-note/train.csv', delimiter=',')
test = np.genfromtxt('bank-note/test.csv', delimiter=',')

pSVM = PrimalSVM(train,
                 test,
                 c,
                 gamma_fnc)

pSVM(100)
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
