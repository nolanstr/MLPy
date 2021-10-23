import numpy as np
from .decision_tree import DecisionTree

class AdaBoost:

    def __init__(self, data, fitness, max_depth, T):

        self.D = np.ones((data[0].shape[0],1)) / data[0].shape[0]

        self.data = data
        self.fitness = fitness
        self.max_depth = max_depth
        self.T = T
        self.trees = []
        self.errors = []
        self.alpha_vals = []
        

    def __call__(self):

        for i in range(self.T):
                    
            self.trees.append(DecisionTree(self.data,
                                           self.fitness,
                                           self.D))

            self.trees[-1](self.max_depth)

            if i == 0:
                self.predictions = self._get_predictions(self.data)
            else:
                self.predictions = np.vstack((self.predictions,
                                              self._get_predictions(self.data)))
            
            self._update_D()
            self._update_error()
            print(i + 1)

    def _update_D(self):

        self._calc_alpha()
        self.D *= np.exp(-self.alpha * self.yh).reshape((self.D.shape[0], 1))

        self.D /= np.sum(self.D)

    def _update_error(self):
        
        self.errors.append(np.count_nonzero(self.yh == 1) / self.yh.shape[0])
        
    def _calc_alpha(self):
        
        self._calc_eta()

        self.alpha = 0.5 * np.log((1-self.eta) / self.eta)
        self.alpha_vals.append(self.alpha)

    def _calc_eta(self):
        
        self._calc_yh()
        
        self.eta = 0.5 - (0.5 * np.sum(self.D.flatten()*self.yh.flatten()))

    def _calc_yh(self):     
        
        self.yh = -1 * np.ones(self.data[0].shape[0])
        
        for i, predict in enumerate(self.predictions[-1]):

            if predict == self.data[1][i]:

                self.yh[i] = 1
            else:
                continue
        
    def _get_predictions(self, data, tree_idx=-1):
        
        predictions = np.zeros((1, data[1].shape[0])).astype('<U13')

        for i, attr in enumerate(data[0]):
            
            predictions[0,i] = self.trees[tree_idx].predict(attr)

        return predictions
            
    def find_adaboost_pred(self, predictions):
        
        ave_predict = np.zeros(predictions.shape).astype('<U13')
        
        for j in range(ave_predict.shape[1]):
            
            vals = np.array(list(set(predictions[:,j])))
            weights = np.zeros(len(vals))

            for i in range(ave_predict.shape[0]):
                 
                idx = np.where(vals == predictions[i,j])
                weights[idx] += self.alpha_vals[i]

                ave_predict[i,j] = vals[np.argmax(weights)]

        return ave_predict 

    def find_all_adaboost_errors(self):

        ave_predict = self.find_adaboost_pred(self.predictions)
        
        errors = np.zeros(ave_predict.shape[0])

        for i in range(errors.shape[0]):
            errors[i] = np.count_nonzero(ave_predict[i] != self.data[1]) /\
                                    self.data[1].shape[0]
        return errors

    def find_test_adaboost_errors(self, data):
        
        predictions = np.zeros(self.predictions.shape).astype('<U13')

        for row in range(predictions.shape[0]):

            predictions[row] = self._get_predictions(data, row)

        ave_predict = self.find_adaboost_pred(predictions)

        errors = np.zeros(ave_predict.shape[0])

        for i in range(errors.shape[0]):
            errors[i] = np.count_nonzero(ave_predict[i] != data[1]) /\
                                    data[1].shape[0]
        return errors

    def find_pure_test_adaboost_errors(self, data):
        
        predictions = np.zeros(self.predictions.shape).astype('<U13')

        for row in range(predictions.shape[0]):

            predictions[row] = self._get_predictions(data, row)

        errors = np.zeros(predictions.shape[0])

        for i in range(errors.shape[0]):
            errors[i] = np.count_nonzero(predictions[i] != data[1]) /\
                                    data[1].shape[0]
        return errors
        
        
    
