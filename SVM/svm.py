import numpy as np
from scipy.optimize import minimize
from numba import jit, njit

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

        self.w_vals = []

    def __call__(self, T, r=0.01):

        np.random.shuffle(self.idxs)

        self.w = np.zeros(self.train_fv.shape[1]).reshape((1,-1))
        self.w_vals.append(self.w.copy())

        for i in range(T):
            
            gamma = self.gamma_fnc(i)

            for xi, yi in zip(self.train_fv[self.idxs,:], 
                                        self.train_labels[self.idxs]):

                pred = self.pred(np.array([xi]))[0]
                
                if pred != yi:

                    self.w[0,0]=0
                    self.w += (-gamma*self.w) + \
                            (gamma*self.C*self.train_fv.shape[0] * yi*xi)
                    self.update_pred_and_errors()

                else:
                    self.w[:,1:] = (1-gamma) * self.w[:,1:]
        
        import pdb;pdb.set_trace() 

    
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


    def __init__(self, train, test, C, gamma_fnc):
        """
        This class performs the primal SVM algorithm.
        self.fv -- feature values
        self.labels -- labels
        """
        
        self.C = C
        self.gamma_fnc = gamma_fnc

        self.train_fv = np.hstack((np.ones((train.shape[0],1)), train[:,0:-1]))
        self.train_labels = train[:,-1].reshape((-1,1))
        self.test_fv = np.hstack((np.ones((test.shape[0],1)), test[:,0:-1]))
        self.test_labels = test[:,-1].reshape((-1,1))
        
        self.train_labels[np.where(self.train_labels == 0)] = -1
        self.test_labels[np.where(self.test_labels == 0)] = -1

    def __call__(self, gamma=0.01, kernel="Linear"):
        
        if kernel == "Linear":
            kernel = self.linear_kernel
        elif kernel == "Gaussian":
            kernel = self.gaussian_kernel
        else:
            print('kernel must be either "Linear" or "Gaussian"')

        bounds = [(0., self.C)] * self.train_fv.shape[0]
        
        constraint = lambda alpha: np.dot(alpha, self.train_labels) 
        constraints_detail = [{"type":"eq", "fun":constraint}] 

        #alpha_guess = np.ones(self.train_fv.shape[0]) * self.C
        alpha_guess = np.full(self.train_fv.shape[0], 1e-5)

        results = minimize(self.dual_obj, 
                              alpha_guess,
                              args=[self.train_fv, self.train_labels,
                                                        kernel, gamma],
                              method='SLSQP',
                              bounds=bounds,
                              constraints=constraints_detail)
        alpha = results['x']
        
        weights = self.compute_weights(alpha, kernel, gamma)
        bias = self.compute_bias_term(weights)
        self.w = weights
        self.update_pred_and_errors()

        import pdb;pdb.set_trace()

    @staticmethod
    def dual_obj(alpha, sup_args):
        
        x_train = sup_args[0]
        y_train = sup_args[1]
        kernel = sup_args[2]
        gamma = sup_args[3]
        alpha = alpha.reshape((-1,1)) 

        term1 = compute_term1(x_train, y_train, alpha, kernel, gamma)

        term2 = np.sum(alpha)

        r_val = (0.5 * term1) - term2

        return r_val

    @staticmethod
    def linear_kernel(x, z, gamma):
        return np.dot(x, z.T)
    
    @staticmethod
    def gaussian_kernel(x, z, gamma):
        return np.exp((-np.linalg.norm(x-z)**2)/c)
    
    def compute_weights(self, alpha, kernel, gamma):
        
        weights = np.sum((alpha * self.train_labels.flatten()).reshape((-1,1))\
                * self.train_fv, axis=0)
        return weights

    def compute_bias_term(self, weights):
        
        b = self.train_labels - np.dot(self.train_fv, weights).reshape((-1,1))

        return b

    def pred(self, fv):
        
        pred = np.sign(np.matmul(fv, self.w.T).flatten())
        pred[np.where(pred == 0)] = -1

        return pred
    
    def update_pred_and_errors(self):
        
        self.train_pred = self.pred(self.train_fv)
        self.test_pred = self.pred(self.test_fv)
        
        train_vals, train_counts = \
                    np.unique(self.train_pred-self.train_labels.flatten(), 
                                                        return_counts=True)
        test_vals, test_counts = \
                    np.unique(self.test_pred-self.test_labels.flatten(),
                                                        return_counts=True)
        
        train_idx = np.where(train_vals == 0)[0][0]
        test_idx = np.where(test_vals == 0)[0][0]
        import pdb;pdb.set_trace()
        self.train_error = 1 - \
                (train_counts[train_idx]/self.train_labels.shape[0])

        self.test_error = 1 - \
                (test_counts[test_idx]/self.test_labels.shape[0])



@jit
def compute_term1(x, y, a, kernel, gamma):
    term1_vals = np.matmul(a, a.T) * np.matmul(y, y.T) * kernel(x, x, gamma)
    return term1_vals.sum()




