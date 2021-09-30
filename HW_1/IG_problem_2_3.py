import numpy as np
import sys
sys.path.append('../../')
sys.path.append('..\..')

from MLPy.DecisionTree.decision_tree import DecisionTree
from MLPy.DecisionTree.fitness_metric.entropy import Entropy


def _gen_clean_data(string):
    train_list = []
    with open(string, 'r') as f:
        for line in f:
            line = line.split(',')
            line[-1] = line[-1][:-1]
            train_list.append(line)
    
    train = np.array(train_list)
    attr = train[:,:-1]
    labels = train[:,-1]
    
    for i in range(attr.shape[1]):
        if np.char.isnumeric(attr[0,i]):
            column = attr[:,i].astype(float)
            med = np.median(column)
            idxs = np.where(column>med)[0]
            attr[idxs, i] = 1
            attr[~idxs, i] = 0
    return attr, labels

train_attr, train_labels = _gen_clean_data('bank/train.csv')
test_attr, test_labels = _gen_clean_data('bank/test.csv')

fitness = Entropy()
tree = DecisionTree(train_attr, train_labels, fitness)
tree(16)
error = []
trees = [tree.create_decision_tree_with_max_depth(i) for i in range(1,17)]
for i, tree_di in enumerate(trees):
#for i, tree_di in enumerate([trees[-2]]):
    print('depth:', i+1)
    cnt = 0
    for i in range(train_attr.shape[0]):
        if tree_di.predict(train_attr[i]) != train_labels[i]:
            cnt += 1
    print('Train Error:', cnt/train_attr.shape[0])
    test_cnt = 0
    for i in range(test_attr.shape[0]):
        if tree_di.predict(test_attr[i]) != test_labels[i]:
            test_cnt += 1
    print('Test Error:', test_cnt/test_attr.shape[0])
    error.append([cnt/train_attr.shape[0], test_cnt/test_attr.shape[0]])

error = np.array(error)
print(np.mean(error, axis=0))
import pdb;pdb.set_trace()
