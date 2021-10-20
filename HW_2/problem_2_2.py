import numpy as np
import sys
sys.path.append('../../')
sys.path.append('..\..')

from MLPy.EnsembleLearning.adaboost import AdaBoost
from MLPy.EnsembleLearning.fitness_metric.entropy import Entropy


train_attr, train_labels = np.load('./bank/clean_train_attr.npy'), \
                            np.load('./bank/clean_train_labels.npy')

fitness = Entropy()
max_depth = 2
T = 5

ab = AdaBoost((train_attr, train_labels), fitness, max_depth, T)
ab()
error = ab.find_all_adaboost_errors()

'''
We need to make it so we can reevaluate adaboost models on test data and not
retrain them.
'''

import pdb;pdb.set_trace()
