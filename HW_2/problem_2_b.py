import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
sys.path.append('..\..')

from MLPy.EnsembleLearning.bagging import Bagging
from MLPy.EnsembleLearning.fitness_metric.entropy import Entropy


train_attr, train_labels = np.load('./bank/clean_train_attr.npy'), \
                            np.load('./bank/clean_train_labels.npy')

test_attr, test_labels = np.load('./bank/clean_test_attr.npy'), \
                            np.load('./bank/clean_test_labels.npy')
fitness = Entropy()
T = 500 

bag = Bagging((train_attr, train_labels), fitness, T)
bag()
error_train = bag.find_all_bagging_errors()
error_test = bag.find_test_bagging_errors((test_attr, test_labels))

plt.plot(np.arange(T), error_train, label='Train Error')
plt.plot(np.arange(T), error_test, label='Test Error')
plt.ylabel('Error')
plt.xlabel('T')

plt.show()

'''
We need to make it so we can reevaluate bagging models on test data and not
retrain them.
'''

import pdb;pdb.set_trace()
