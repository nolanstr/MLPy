import numpy as np
from .decision_tree import DecisionTree

class RandomTree:

    def __init__(self, data, fitness, T, subset_size):

        self.data = data
        self.fitness = fitness
        self.T = T
        self.trees = []
        self.errors = []
        self.alpha_vals = []
        self.m = self.data[0].shape[0]
        self.subset_size = subset_size

    def __call__(self):

        for i in range(self.T):
            
            idxs = np.random.randint(low=0, high=self.m, size=self.m)
                    
            rnd_data = (self.data[0][idxs,:], self.data[1][idxs])
            
            attr_subset = np.random.choice(self.data[0].shape[1],
                                                self.subset_size,
                                                replace=False)

            self.trees.append(DecisionTree(rnd_data,
                                           self.fitness,
                                           attr_subset=attr_subset))

            self.trees[-1]()

            if i == 0:
                self.predictions = self._get_predictions(self.data)
            else:
                self.predictions = np.vstack((self.predictions,
                                              self._get_predictions(self.data)))
            
            #import pdb;pdb.set_trace()
            self._update_error()
            #import pdb;pdb.set_trace()
            print(i + 1)

    def _update_error(self):
        
        self._calc_yh()
        self.errors.append(np.count_nonzero(self.yh == 1) / self.yh.shape[0])
        
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
            
    def find_random_tree_pred(self, predictions):
        
        ave_predict = np.zeros(predictions.shape).astype('<U13')
        
        for j in range(ave_predict.shape[1]):
            
            vals = np.unique(predictions[:,j])
            counts = np.zeros(len(vals))

            for i in range(ave_predict.shape[0]):
                 
                idx = np.where(vals == predictions[i,j])
                counts[idx] += 1

                ave_predict[i,j] = vals[np.argmax(counts)]

        return ave_predict 

    def find_all_random_tree_errors(self):

        ave_predict = self.find_random_tree_pred(self.predictions)
        
        errors = np.zeros(ave_predict.shape[0])

        for i in range(errors.shape[0]):
            errors[i] = np.count_nonzero(ave_predict[i] != self.data[1]) /\
                                    self.data[1].shape[0]
        return errors

    def find_test_random_tree_errors(self, data):
        
        predictions = np.zeros(self.predictions.shape).astype('<U13')

        for row in range(predictions.shape[0]):

            predictions[row] = self._get_predictions(data, row)

        ave_predict = self.find_random_tree_pred(predictions)

        errors = np.zeros(ave_predict.shape[0])

        for i in range(errors.shape[0]):
            errors[i] = np.count_nonzero(ave_predict[i] != data[1]) /\
                                    data[1].shape[0]
        return errors
        
        
    
