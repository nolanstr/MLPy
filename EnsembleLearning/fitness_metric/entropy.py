import numpy as np

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
        
        expected_entropy = self._entropy(labels, D)
        expected_entropies = self._expected_entropy(attributes, labels, D)
        information_gain = np.ones(len(expected_entropies))*\
                                        expected_entropy - expected_entropies

        return information_gain

        
    def _expected_entropy(self, attributes, labels, D):
        
        entropies = []

        for idx in range(attributes.shape[1]):
            
            values = set(attributes[:,idx].tolist())
            entropy = 0

            for value in values:
                #focus on updates here
                value_idxs = np.where(attributes[:,idx] == value)[0]
                entropy += (value_idxs.shape[0] / labels.shape[0])\
                                        * self._entropy(labels[value_idxs], 
                                                        D[value_idxs])
            entropies.append(entropy)

        return entropies

    def _entropy(self, labels, D):
        
        options = set(labels.tolist())
        label_probs = [np.sum(D[np.where(labels==option)])/np.sum(D) for \
                                                    option in options]
        entropy = sum([-prob*np.log2(prob) for prob in label_probs])
        
        return entropy

