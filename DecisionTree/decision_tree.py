from itertools import chain

from .branch import Branch

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

    def _create_tree_dict(self):
        '''
        Initializes tree dictionary for that is continually updated.
        '''

        self.tree_dict = {}
        self.tree_dict[str(self.branches[-1][0].path[0])] = {}

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
             
            idx = int([key for key in int_dict.keys()][0])
            int_dict = int_dict[str(idx)]
            val = attributes[idx]
            
            if isinstance(int_dict[str(val)], dict):
                int_dict = int_dict[str(val)]
            else:
                return int_dict[str(val)]





