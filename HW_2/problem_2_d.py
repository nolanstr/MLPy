import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
sys.path.append('..\..')

from MLPy.EnsembleLearning.random_tree import RandomTree
from MLPy.EnsembleLearning.fitness_metric.entropy import Entropy


train_attr, train_labels = np.load('./bank/clean_train_attr.npy'), \
                            np.load('./bank/clean_train_labels.npy')

test_attr, test_labels = np.load('./bank/clean_test_attr.npy'), \
                            np.load('./bank/clean_test_labels.npy')
fitness = Entropy()
colors = ['r','b','g']

for i, ss in enumerate([2,4,6]):
    T = 200 
    subset_size = ss

    rt = RandomTree((train_attr, train_labels), fitness, T, subset_size)
    rt()
    error_train = rt.find_all_random_tree_errors()
    error_test = rt.find_test_random_tree_errors((test_attr, test_labels))

    plt.plot(np.arange(T), error_train, colors[i], label='Train Error')
    plt.plot(np.arange(T), error_test, colors[i], label='Test Error')
    plt.ylabel('Error')
    plt.xlabel('T')

plt.savefig('RandomTree'+str(subset_size)+'.png')

'''
We need to make it so we can reevaluate RandomTree models on test data and not
retrain them.
'''

import pdb;pdb.set_trace()
