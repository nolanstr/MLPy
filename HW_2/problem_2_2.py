import numpy as np
import sys
sys.path.append('../../')
sys.path.append('..\..')

from MLPy.EnsembleLearning.adaboost import AdaBoost
from MLPy.EnsembleLearning.fitness_metric.entropy import Entropy


train_attr, train_labels = np.load('bank/clean_train_attr.csv'), \
                            np.load('bank/clean_train_labels.csv')

fitness = Entropy()
ab = AdaBoost((train_attr, train_labels), fitness, 2, 500)
ab()


print(np.mean(error, axis=0))
import pdb;pdb.set_trace()
