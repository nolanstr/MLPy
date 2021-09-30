import numpy as np
import sys
sys.path.append('../../')
sys.path.append('..\..')

from MLPy.DecisionTree.decision_tree import DecisionTree
from MLPy.DecisionTree.fitness_metric.entropy import Entropy
from MLPy.DecisionTree.fitness_metric.majority_error import MajorityError 

attr = np.array([[0,0,1,0],
                 [0,1,0,0],
                 [0,0,1,1],
                 [1,0,0,1],
                 [0,1,1,0],
                 [1,1,0,0],
                 [0,1,0,1]])
labels = np.array([[0],
                   [0], 
                   [1],
                   [1],
                   [0],
                   [0],
                   [0]])

import pdb;pdb.set_trace()
fitness = MajorityError()
tree = DecisionTree(attr, labels, fitness)
tree()
import pdb;pdb.set_trace()
