import numpy as np
from itertools import chain

from .branch import Branch
from copy import deepcopy

class DecisionTree:

    def __init__(self, attributes, labels, fitness):
        '''
        attributes --> shape=(n,m)
        labels --> shape=(n,1)
        '''
        try:
            labels = labels.flatten()
        except:
            pass
        self.attributes = attributes
        self.labels = labels
        self.branches = [[Branch([], [], attributes, labels)]]
        self.fitness = fitness

    def __call__(self, max_depth=None):
        '''
        Splits the data using Branch class and continually calling each branch
        instance to split data and record decisions by making new instancees of
        branches.
        '''
        self.branches.append(self.branches[0][0].split(self.fitness, max_depth))
        self._create_tree_dict()

        while self._check_branches():
            
            next_level = []

            for branch in self.branches[-1]:
                
                next_level += branch.split(self.fitness, max_depth)

            self.branches.append(next_level)
            self._update_tree_dict(self.branches[-1])
   
    def create_decision_tree_with_max_depth(self, max_depth):

        for i, bunch in enumerate(self.branches):

            depths = np.array([branch.depth for branch in bunch])
            
            if np.all(depths <= max_depth):
                continue
            else:
                cp = deepcopy(self)
                new_dt = DecisionTree(cp.attributes, cp.labels, cp.fitness)
                new_dt.branches = [cp.branches[i] for i in range(i)]
                new_dt._create_tree_dict(1)
                new_dt._update_tree_dict(new_dt.branches[-1])
                
                return new_dt
        return deepcopy(self)

    def _update_tree_dict(self, next_branch):
        '''
        updates tree dictionary given 
        '''

        for branch in next_branch:
            
            dict_path = self._create_dict_path(branch)

            for i, key_val in enumerate(dict_path):
                
                if i == 0:
                    int_dict = self.tree_dict[str(key_val)]
                else:
                    if branch._leaf and i == len(dict_path)-1:
                        int_dict[str(key_val)] = branch._leaf_value
                    elif key_val in int_dict.keys():
                        int_dict = int_dict[str(key_val)]
                    else:
                        int_dict[str(key_val)] = {}
                        int_dict = int_dict[str(key_val)]
                    
            int_dict['MCV'] = branch._leaf_value
                    #if i == len(dict_path)-1 and not branch._leaf:
                    #    int_dict['MCV'] = branch._leaf_value

    def _create_tree_dict(self, i=-1):
        '''
        Initializes tree dictionary for that is continually updated.
        '''

        self.tree_dict = {}
        self.tree_dict[str(self.branches[i][0].path[0])] = {}
    
    def _create_dict_path(self, branch):
        '''
        combines path and label lists in form [path_0, label_0, ..., path_n,
        label_n]
        '''

        dict_path = list(chain.from_iterable((idx,label) for idx,label\
                                        in zip(branch.path, branch.choices)))
        dict_path = [str(val) for val in dict_path]
        return dict_path

    def _check_branches(self):
        '''
        Checks to see if all branches have turned into leaf nodes.
        '''

        if len(self.branches) == 1:
            return True
        
        for branch in self.branches[-1]:
            if not branch._leaf:
                return True

        return False
    
    def predict(self, attributes):
        
        int_dict = self.tree_dict
        
        while True:
            
            idxs = np.array([key for key in int_dict.keys()])
            idx = idxs[np.char.isnumeric(idxs)]

            try:
                if idx[0] not in int_dict.keys():
                    return int_dict['MCV']
            except:
                pass
            if len(idx) == 0:
                return int_dict['MCV']
            val = attributes[int(idx)]
            
            if val not in int_dict[idx[0]].keys(): 
                try:
                    return prev_dict['MCV']
                except:
                    pass
            int_dict = int_dict[idx[0]]
             
            try:
                if isinstance(int_dict[str(val)], dict):
                    prev_dict = int_dict
                    int_dict = int_dict[str(val)]
                else:
                    return int_dict[str(val)]
            except:
                return int_dict['MCV']
