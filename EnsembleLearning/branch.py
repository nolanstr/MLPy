import numpy as np
from copy import deepcopy

class Branch:

    def __init__(self, path, choices, attributes, labels, D, subset_size, 
                                            a_idxs=None):
        

        if a_idxs is None:
            a_idxs = np.arange(attributes.shape[1])

        self.path = path
        self.choices = choices
        self.attributes = attributes
        self.labels = labels
        self.D = D 
        self._leaf = False
        self.subset_size = subset_size
        self.a_idxs = a_idxs
        try:
            rnd_idxs = np.random.choice(self.a_idxs, self.subset_size, 
                                                    replace=False)
        except:
            rnd_idxs = False

        self.rnd_idxs = rnd_idxs
        
        self.depth = len(self.path)
        self._check_for_leaf(None)
        
    def split(self, fitness, max_depth):

        if self._check_for_leaf(max_depth) or self.rnd_idxs is False:
            return [deepcopy(self)]
        
        fitness_eval = fitness(self.attributes, self.labels, self.D)
        try:
            ranked_rnd_idxs = np.flip(
                    self.rnd_idxs[np.argsort(fitness_eval[self.rnd_idxs])])
        except:
            import pdb;pdb.set_trace()

        split_attr_idx = ranked_rnd_idxs[0] 

        if split_attr_idx in self.path:
            
            for index in ranked_rnd_idxs:
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
        new_a_idxs = self.a_idxs[self.a_idxs != split_attr_idx]

        for val in pos_values:
            
            idxs = np.where(self.attributes[:,split_attr_idx] == val)[0]

            new_branches.append(Branch(path + [split_attr_idx],
                                       choices + [val],
                                       c_self.attributes[idxs,:], 
                                       c_self.labels[idxs],
                                       c_self.D[idxs],
                                       c_self.subset_size,
                                       a_idxs=new_a_idxs))
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
            weights = np.zeros(vals.shape[0])
            
            for i, val in enumerate(vals):
                
                weights[i] += np.sum(self.D[np.where(self.labels==val)])

            self._leaf_value = vals[np.argmax(weights)]

            return True

        vals, counts = np.unique(self.labels, return_counts=True)
        self._leaf_value = vals[np.argmax(counts)]

        return False
