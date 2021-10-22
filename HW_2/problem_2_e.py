import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
sys.path.append('..\..')

from MLPy.EnsembleLearning.random_tree import RandomTree
from MLPy.EnsembleLearning.fitness_metric.entropy import Entropy



def bridge(seed):
    
    train_attr, train_labels = np.load('./bank/clean_train_attr.npy'), \
                                np.load('./bank/clean_train_labels.npy')

    test_attr, test_labels = np.load('./bank/clean_test_attr.npy'), \
                                np.load('./bank/clean_test_labels.npy')
    fitness = Entropy()
    T = 200
    np.random.seed(seed)
    idxs = np.random.randint(low=0, high=train_attr.shape[0], size=1000)
    rt = RandomTree((train_attr[idxs,:], train_labels[idxs]), fitness, T, 4)
    rt()
    #error_train = bag.find_all_RandomTree_errors()
    #error_test = bag.find_test_RandomTree_errors((test_attr[idxs,:], 
    #                                                test_labels[idxs]))
    test_data = (test_attr, test_labels)

    pred = np.zeros((T, test_labels.shape[0])).astype('<U13')
    for i in range(T):
        
        pred[i] = rt._get_predictions(test_data, i)
        print(np.unique(pred[i]))
    bin_pred = np.copy(pred)
    
    bin_pred[bin_pred == 'yes'] = 1
    bin_pred[bin_pred == 'None'] = 0
    bin_pred[bin_pred == 'no'] = 0
    
    #print(np.unique(bin_pred))
    bin_pred = bin_pred.astype(float)

    return bin_pred
    

from multiprocessing import Pool

n_folds = 100
seeds = np.random.choice(np.arange(50,100), 30, replace=False)
cores = 30
p = Pool(cores)
pred = p.map(bridge, seeds)

all_pred = np.array(pred)
test_labels = np.load('./bank/clean_test_labels.npy')
test_labels[test_labels == 'yes'] = 1
test_labels[test_labels == 'no'] = 0
test_labels = test_labels.astype(float)

bias_terms = []
sv_terms = []

for i in range(all_pred.shape[0]):
    tree_i = all_pred[:,i,:]

    bias = np.mean((np.mean(tree_i, axis=0) - test_labels)**2)
    sv = np.mean((1/(tree_i.shape[0] - 1)) * \
            np.sum((tree_i - test_labels)**2, axis=0))
    bias_terms.append(bias)
    sv_terms.append(sv)
#np.save('part_c_data_30f_200T.npy', np.array(error))
#bridge(10)
print('Single tree bias:', bias_terms[0])
print('Single tree sv:', sv_terms[0])
print('All trees bias:', np.mean(bias_terms))
print('All trees sv:', np.mean(sv_terms))

import pdb;pdb.set_trace()
