import numpy as np
from copy import deepcopy

class Branch:

    def __init__(self, path, choices, attributes, labels, D, attr_subset): 

        self.path = path
        self.choices = choices
        self.attributes = attributes
        self.labels = labels
        self.D = D 
        self._leaf = False
        self.attr_subset = attr_subset

        self.depth = len(self.path)
        self._check_for_leaf(None)
        
    def split(self, fitness, max_depth):

        if self._check_for_leaf(max_depth):
            return [deepcopy(self)]
        
        fitness_eval = fitness(self.attributes, self.labels, self.D)
        allowable_idxs = np.array(list(set(self.attr_subset) - set(self.path)))
        split_attr_idx = allowable_idxs[
                    np.argmax(fitness_eval[allowable_idxs])] 

        pos_values = list(set(self.attributes[:,split_attr_idx].tolist()))
        
        c_self = deepcopy(self)
        path = c_self.path
        choices = c_self.choices
        new_branches = []

        for val in pos_values:
            
            idxs = np.where(self.attributes[:,split_attr_idx] == val)[0]

            new_branches.append(Branch(path + [split_attr_idx],
                                       choices + [val],
                                       c_self.attributes[idxs,:], 
                                       c_self.labels[idxs],
                                       c_self.D[idxs],
                                       attr_subset=c_self.attr_subset))
        return new_branches

    def _check_for_leaf(self, max_depth):
         
        if np.all(self.labels == self.labels[0]):
            self._leaf = True
            self._leaf_value = self.labels[0]
            return True
        
        if np.all(self.attributes == self.attributes[0,:]) \
                or set(self.attr_subset) == set(self.path):
            self._leaf = True
            vals, counts = np.unique(self.labels, return_counts=True)
            self._leaf_value = vals[np.argmax(counts)]
            return True

        if self.depth == max_depth:
            self._leaf = True
            vals, counts = np.unique(self.labels, return_counts=True)
            weights = np.zeros(vals.shape[0])
            
            for i, val in enumerate(vals):
                
                weights[i] += np.sum(self.D[np.where(self.labels==val)])

            self._leaf_value = vals[np.argmax(weights)]

            return True

        vals, counts = np.unique(self.labels, return_counts=True)
        self._leaf_value = vals[np.argmax(counts)]

        return False
