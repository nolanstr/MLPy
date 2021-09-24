import numpy as np

class MajorityError:
    '''
    Initializes instance of self that can compute information gained based
    and will generate respective decision tree.
    '''
        
    def __call__(self, attributes, labels):
         
        information_gain = self._information_gain(attributes, labels)

        return np.argmax(information_gain)

    def _information_gain(self, attributes, labels):
        '''
        Takes current entropy and returns column idx for next split
        '''
        
        expected_majority_error = self._entropy(labels)
        expected_majority_errors = self._expected_entropy(attributes, labels)
        information_gain = np.ones(len(expected_majority_errors))*\
                        expected_majority_error - expected_majority_errors

        return information_gain

        
    def _expected_majority_error(self, attributes, labels):
        
        majority_errors = []

        for idx in range(attributes.shape[1]):
            
            values = set(attributes[:,idx].tolist())
            majority_error = 0

            for value in values:

                value_idxs = np.where(attributes[:,idx] == value)[0]
                majority_error += (value_idxs.shape[0] / labels.shape[0])\
                                    * self._majority_error(labels[value_idxs])
            
            majority_errors.append(entropy)

        return entropies

    def _majority_error(self, labels):
        
        options = set(labels.tolist())
        label_probs = [np.count_nonzero(labels==option)/labels.shape[0] for \
                                                            option in options]
        majority_error = sum([prob*min(prob) for prob in label_probs])
        
        return majority_error

