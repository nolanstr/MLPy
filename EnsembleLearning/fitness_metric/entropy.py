import numpy as np
from numba import jit

class Entropy:
    '''
    Initializes instance of self that can compute information gained based
    and will generate respective decision tree.
    '''

        
    def __call__(self, attributes, labels, D):
         
        information_gain = self._information_gain(attributes, labels, D)

        return information_gain

    def _information_gain(self, attributes, labels, D):
        '''
        Takes current entropy and returns column idx for next split
        '''
        
        expected_entropy = _entropy(labels, D)
        expected_entropies = self._expected_entropy(attributes, labels, D)
        information_gain = expected_entropy - expected_entropies
        
        return information_gain

        
    def _expected_entropy(self, attributes, labels, D):
        
        entropies = []

        for idx in range(attributes.shape[1]):
            
            values = np.unique(attributes[:,idx])
            entropy = 0

            for value in values:

                value_idxs = np.where(attributes[:,idx] == value)[0]
                #entropy += (np.sum(D[value_idxs]) / np.sum(D)) \
                #                        * _entropy(labels[value_idxs], 
                #                                        D[value_idxs])
                entropy += (value_idxs.shape[0] / D.shape[0]) \
                                        * _entropy(labels[value_idxs], 
                                                        D[value_idxs])
            entropies.append(entropy)

        return entropies

#@jit
def _entropy(labels, D):
    
    options = np.unique(labels)
    idxs = [np.where(labels==option) for option in options]
    
    D_norm = np.sum(D)

    label_probs = np.array([D[idx].sum() / D_norm for idx in  idxs])
    entropy = np.sum(-label_probs*np.log2(label_probs))
    
    return entropy

