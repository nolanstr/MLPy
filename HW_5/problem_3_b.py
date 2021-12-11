import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')

from MLPy.LogisticRegression.logistic_regression import LogisticRegression

train_data = np.genfromtxt('bank-note/train.csv', delimiter=',')
x_train = train_data[:,0:-1]
x_train = np.hstack((np.ones(train_data.shape[0]).reshape((-1,1)),
                                                train_data[:,0:-1]))
y_train = train_data[:,-1].flatten()
y_train[np.where(y_train == 0)] = -1

test_data = np.genfromtxt('bank-note/test.csv', delimiter=',')
x_test = np.hstack((np.ones(test_data.shape[0]).reshape((-1,1)), 
                                                test_data[:,0:-1]))
y_test = test_data[:,-1].flatten()
y_test[np.where(y_test == 0)] = -1

gamma_0 = 1e-1
d = 1e0
gamma = lambda t: gamma_0 / (1 + (gamma_0*t/d))

variances = np.array([0.01, 0.1, 0.5, 1, 3, 5, 10, 100])
stds = np.sqrt(variances)

LRs = []

for std in [0]:
    
    LRs.append(LogisticRegression(x_train, y_train, gamma, method='MLE'))
    errors = LRs[-1].get_test_train_error(x_test, y_test)

    LRs[-1]()
    
    errors = LRs[-1].get_test_train_error(x_test, y_test)
    
    print(f'Train Error: {errors[0]*100} \n Test Error:{errors[1]*100}')
    
    #plt.plot(np.arange(len(LRs[-1].js)), LRs[-1].js)
    #plt.xlabel('Updates')
    #plt.ylabel('J')
    
    plt.show()
    plt.clf()


