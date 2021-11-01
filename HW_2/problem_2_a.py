import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
sys.path.append('..\..')

from MLPy.EnsembleLearning.adaboost import AdaBoost
from MLPy.EnsembleLearning.fitness_metric.entropy import Entropy


train_attr, train_labels = np.load('./bank/clean_train_attr.npy'), \
                            np.load('./bank/clean_train_labels.npy')

test_attr, test_labels = np.load('./bank/clean_test_attr.npy'), \
                            np.load('./bank/clean_test_labels.npy')
fitness = Entropy()
max_depth = 1
T = 500

ab = AdaBoost((train_attr, train_labels), fitness, max_depth, T)
ab()
error_train = ab.find_all_adaboost_errors()
error_test = ab.find_test_adaboost_errors((test_attr, test_labels))

plt.plot(np.arange(T), ab.errors, label='Train Error')
plt.plot(np.arange(T), ab.find_pure_test_adaboost_errors((test_attr, test_labels)), 
                                        label='Test Error')
plt.ylabel('Error')
plt.xlabel('T')

plt.show()

'''
We need to make it so we can reevaluate adaboost models on test data and not
retrain them.
'''

import pdb;pdb.set_trace()
