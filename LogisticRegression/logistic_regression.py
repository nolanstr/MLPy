import numpy as np

import sys
sys.path.append('../../')

from MLPy.LogisticRegression.lr_methods import *

methods = {'MAP':MAP, 
           'MLE':MLE}

class LogisticRegression:

    def __init__(self, x, y, gamma, std=None, method='MLE'):

        self.x = x
        self.y = y

        if method.upper() == 'MAP':
            self.w = np.random.normal(loc=0.0, scale=std, size=x.shape[1])
        else:
            self.w = np.random.normal(loc=0.0, scale=1, size=x.shape[1])

        self.idxs = np.arange(self.x.shape[0])

        self.method = methods[method.upper()](std)

        self.gamma = gamma

        self.djdws = []
        self.js = [] 

    def __call__(self, T=100):
        
        for t in range(T):
            
            np.random.shuffle(self.idxs)

            for (xi,yi) in zip(self.x[self.idxs,:], self.y[self.idxs]):
                dJdW = self.method.dJdW(xi.flatten(),yi,self.w,self.x.shape[0])
                self.w -= self.gamma(t) * dJdW
                self.djdws.append(dJdW)
                self.js.append(self.method.J(xi.flatten(),yi,self.w))

    def get_test_train_error(self, x_test, y_test):

        y_train_approx = np.sign(self.pred(self.x))
        y_test_approx = np.sign(self.pred(x_test))
         
        train_error = np.count_nonzero(y_train_approx - self.y) /\
                                                        self.y.shape[0]
        test_error = np.count_nonzero(y_test_approx - y_test) /\
                                y_test.shape[0]
        
        return train_error, test_error
    
    def pred(self, x):

        pred_y = np.ones(x.shape[0]) * 1

        for i in range(x.shape[0]):

            pred_y[i] = np.sign(np.dot(self.w, x[i].flatten()))

        return pred_y
