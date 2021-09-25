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
        
        expected_gini_error = self._gini_idx(labels)
        expected_gini_errors = self._expected_gini_error(attributes, labels)
        information_gain = np.ones(len(expected_gini_errors))*\
                        expected_gini_error - expected_gini_errors

        return information_gain

        
    def _expected_gini_error(self, attributes, labels):
        
        gini_errors = []

        for idx in range(attributes.shape[1]):
            
            values = set(attributes[:,idx].tolist())
            gini_error = 0

            for value in values:

                value_idxs = np.where(attributes[:,idx] == value)[0]
                gini_error += (value_idxs.shape[0] / labels.shape[0])\
                                    * self._gini_idx(labels[value_idxs])
            
            gini_errors.append(gini_error)

        return gini_errors

    def _gini_idx(self, labels):
        
        options = set(labels.tolist())
        label_probs = [np.count_nonzero(labels==option)/labels.shape[0] for \
                                                            option in options]

        if len(label_probs) == 1:
            return 0
        else:
            return min(label_probs) 
