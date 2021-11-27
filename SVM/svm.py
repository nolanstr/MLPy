import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from numba import jit, njit, uint64

class PrimalSVM:

    def __init__(self, train, test, C, gamma_fnc):
        """
        This class performs the primal SVM algorithm.
        self.fv -- feature values
        self.labels -- labels
        """
        
        self.C = C
        self.gamma_fnc = gamma_fnc
        
        self.train_fv = np.hstack((np.ones((train.shape[0],1)), train[:,0:-1]))
        self.train_labels = train[:,-1]
        self.test_fv = np.hstack((np.ones((test.shape[0],1)), test[:,0:-1]))
        self.test_labels = test[:,-1]
        
        self.train_labels[np.where(self.train_labels == 0)] = -1
        self.test_labels[np.where(self.test_labels == 0)] = -1

        self.idxs = np.arange(self.train_fv.shape[0])
        
        self.train_error = []
        self.test_error = []
        self.J_vals = []

        self.w_vals = []

    def __call__(self, T, r=0.01):

        np.random.shuffle(self.idxs)

        self.w = np.zeros(self.train_fv.shape[1]).reshape((1,-1))
        self.w_vals.append(self.w.copy())

        for i in range(T):
            
            gamma = self.gamma_fnc(i)
            self.update_pred_and_errors()

            for xi, yi in zip(self.train_fv[self.idxs,:], 
                                        self.train_labels[self.idxs]):
                

                pred = self.pred(np.array([xi]))[0]
                self.update_J()
                if pred != yi:

                    self.w -= (gamma*np.append([0], self.w[1:])) +- \
                            (gamma * self.C * self.train_fv.shape[0] * yi*xi)
                    self.update_pred_and_errors()

                else:
                    self.w[:,1:] = (1-gamma) * self.w[:,1:]

        self.update_J(True) 
    
    def update_J(self, check=False):
        
        adjusted_pred = 1 - (self.train_labels * self.train_pred)
        check_idxs = np.where(adjusted_pred <= 0)
        adjusted_pred[check_idxs] = 0
        J_w = (0.5 * np.dot(self.w[1:].flatten(),self.w[1:].flatten())) \
                                        + self.C * np.sum(adjusted_pred)
        
        self.J_vals.append(J_w)

    def pred(self, fv):
        
        pred = np.sign(np.matmul(fv, self.w.T).flatten())
        pred[np.where(pred == 0)] = -1

        return pred
    
    def update_pred_and_errors(self):
        
        self.train_pred = self.pred(self.train_fv)
        self.test_pred = self.pred(self.test_fv)
        
        _, train_counts = np.unique(self.train_pred*self.train_labels, 
                                                    return_counts=True)
        _, test_counts = np.unique(self.test_pred*self.test_labels, 
                                                    return_counts=True)
        
        self.train_error.append(train_counts[0]/self.train_labels.shape[0])
        self.test_error.append(test_counts[0]/self.test_labels.shape[0])


class DualSVM:


    def __init__(self, train, test, C):
        """
        This class performs the primal SVM algorithm.
        self.fv -- feature values
        self.labels -- labels
        """
        
        self.C = C

        self.train_fv = train[:,0:-1]
        self.train_labels = train[:,-1].reshape((-1,1))
        self.test_fv = test[:,0:-1]
        self.test_labels = test[:,-1].reshape((-1,1))
        
        self.train_labels[np.where(self.train_labels == 0)] = -1
        self.test_labels[np.where(self.test_labels == 0)] = -1

    def __call__(self, gamma=0.01, kernel="Linear"):
        

        self.kernel_name = kernel
        self.gamma = gamma

        if kernel == "Linear":
            self.kernel = self.linear_kernel
        elif kernel == "Gaussian":
            self.kernel_val = self.gaussian_kernel(self.train_fv, 
                                                   self.train_fv, 
                                                   gamma)
            self.kernel = self.gaussian_kernel

        else:
            print('kernel must be either "Linear" or "Gaussian"')

        bounds = [(0., self.C)] * self.train_fv.shape[0]
        
        constraint = lambda alpha: np.dot(alpha, self.train_labels) 
        constraints_detail = [{"type":"eq", "fun":constraint}] 

        alpha_guess = np.full(self.train_fv.shape[0], 0)

        results = minimize(dual_obj, 
                              alpha_guess,
                              args=[self.train_fv, self.train_labels,
                                                        self.kernel, gamma],
                              method='SLSQP',
                              bounds=bounds,
                              constraints=constraints_detail)

        self.alpha = results['x']
        self.alpha[np.isclose(self.alpha, self.C)] = self.C
        self.alpha[np.isclose(self.alpha, 0)] = 0

        self.w = self.compute_weights(self.alpha, kernel, gamma)
        self.bias = self.compute_bias_term(self.w)
        self.update_pred_and_errors()


    @staticmethod
    def linear_kernel(x, z, gamma):
        return np.dot(x, z.T)
    

    def gaussian_kernel(self, x, z, gamma, check=True):
        
        if check:
            return np.exp(-cdist(x,z,'sqeuclidean')/gamma) 

        else:
            return self.kernel_val

    def compute_weights(self, alpha, kernel, gamma):
        
        weights = np.sum((alpha * self.train_labels.flatten()).reshape((-1,1))\
                * self.train_fv, axis=0)
        return weights

    def compute_bias_term(self, weights):
        
        b = self.train_labels - np.dot(self.train_fv, weights).reshape((-1,1))

        return np.mean(b)

    def pred(self, fv, labels, Z=None):
        
        if self.kernel_name == 'Linear':
            pred = np.sign(np.matmul(fv, self.w.T).flatten() + self.bias)
            pred[np.where(pred == 0)] = -1

            return pred
        else:
            if Z is None:
                Z = fv
            pred = np.sign(np.sum((
                        self.alpha.flatten()*labels.flatten()).reshape(-1,1)\
                        *self.kernel(fv,Z,self.gamma), axis=0))

            return pred

    def update_pred_and_errors(self):
        self.train_pred = self.pred(self.train_fv, self.train_labels)
        self.test_pred = self.pred(self.train_fv, self.train_labels, 
                                                        self.test_fv)
        
        train_vals, train_counts = \
                    np.unique(self.train_pred-self.train_labels.flatten(), 
                                                        return_counts=True)
        test_vals, test_counts = \
                    np.unique(self.test_pred-self.test_labels.flatten(),
                                                        return_counts=True)
        
        train_idx = np.where(train_vals == 0)[0][0]
        test_idx = np.where(test_vals == 0)[0][0]
        self.train_error = 1 - \
                (train_counts[train_idx]/self.train_labels.shape[0])

        self.test_error = 1 - \
                (test_counts[test_idx]/self.test_labels.shape[0])



@jit
def dual_obj(alpha, sup_args):
    
    x = sup_args[0]
    y = sup_args[1]
    kernel = sup_args[2]
    gamma = sup_args[3]
    alpha = alpha.reshape((-1,1)) 
    term1 = np.sum(np.outer(alpha, alpha) * 
                np.outer(y, y) * kernel(x,x,gamma,False))
    term2 = np.sum(alpha)

    r_val = (0.5 * term1) - term2
    return r_val
