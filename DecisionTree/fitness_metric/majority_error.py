import numpy as np

class MajorityError:
    '''
    Initializes instance of self that can compute information gained based
    and will generate respective decision tree.
    '''
        
    def __call__(self, attributes, labels):
         
        information_gain = self._information_gain(attributes, labels)
        
        return information_gain

    def _information_gain(self, attributes, labels):
        '''
        Takes current entropy and returns column idx for next split
        '''
        
        expected_majority_error = self._majority_error(labels, True)
        expected_majority_errors = self._expected_majority_error(attributes, labels)
        information_gain = np.ones(len(expected_majority_errors))*\
                        expected_majority_error - expected_majority_errors
        return abs(information_gain)

        
    def _expected_majority_error(self, attributes, labels):
        
        majority_errors = []

        for idx in range(attributes.shape[1]):
            
            values = list(set(attributes[:,idx].tolist()))
            majority_error = 0

            for value in values:
                value_idxs = np.where(attributes[:,idx] == value)[0]
                majority_error += self._majority_error(labels, value_idxs)

            
            majority_errors.append(majority_error)

        return majority_errors

    def _majority_error(self, labels, idxs):
        
        options = set(labels.tolist())
        unique, counts = np.unique(labels[idxs], return_counts=True)
        min_label_prob = counts.min() / counts.sum()
        if min_label_prob == 1:
            min_label_prob = 0.0
        set_prob = labels[idxs].flatten().shape[0] / labels.shape[0] 
        majority_error = set_prob * min_label_prob
        return majority_error 
