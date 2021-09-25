import numpy as np
from copy import deepcopy

class Branch:

    def __init__(self, path, choices, attributes, labels):

        self.path = path
        self.choices = choices
        self.attributes = attributes
        self.labels = labels
         
        self._leaf = False

        self.depth = len(self.path)
        self._check_for_leaf(None)
        
    def split(self, fitness, max_depth):
        if self._check_for_leaf(max_depth):
            return [deepcopy(self)]
        
        
        fitness_eval = fitness(self.attributes, self.labels)
        split_attr_idx = np.argmax(fitness_eval)
        
        if split_attr_idx in self.path:
            sort_hi_low = np.flip(np.argsort(fitness_eval))
            for index in sort_hi_low:
                if index in self.path:
                    continue
                else:
                    split_attr_idx = index
                    break
            if split_attr_idx in self.path:
                self._leaf = True


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
                                       c_self.labels[idxs]))
        return new_branches

    def _check_for_leaf(self, max_depth):
         
        if np.all(self.labels == self.labels[0]):

            self._leaf = True
            self._leaf_value = self.labels[0]
            return True

        if np.all(self.attributes == self.attributes[0,:]):
            self._leaf = True
            vals, counts = np.unique(self.labels, return_counts=True)
            self._leaf_value = vals[np.argmax(counts)]
            return True

        if self.depth == max_depth:
            self._leaf = True
            vals, counts = np.unique(self.labels, return_counts=True)
            self._leaf_value = vals[np.argmax(counts)]
            return True
        vals, counts = np.unique(self.labels, return_counts=True)
        self._leaf_value = vals[np.argmax(counts)]

        return False
