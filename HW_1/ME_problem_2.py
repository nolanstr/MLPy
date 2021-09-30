import numpy as np
import sys
sys.path.append('../../')
sys.path.append('..\..')

from MLPy.DecisionTree.decision_tree import DecisionTree
from MLPy.DecisionTree.fitness_metric.majority_error import MajorityError

train_car_list = []

with open('car/train.csv', 'r') as f:
    for line in f:
        line = line.split(',')
        line[-1] = line[-1][:-1]
        train_car_list.append(line)


train_car = np.array(train_car_list)
attr = train_car[:,:-1]
labels = train_car[:,-1]

fitness = MajorityError()
tree = DecisionTree(attr, labels, fitness)
tree()#6)
trees = [tree.create_decision_tree_with_max_depth(i) for i in range(1,7)]

test_car_list = []

with open('car/test.csv', 'r') as f:
    for line in f:
        line = line.split(',')
        line[-1] = line[-1][:-1]
        test_car_list.append(line)


test_car = np.array(test_car_list)
test_attr = test_car[:,:-1]
test_labels = test_car[:,-1]

import pdb;pdb.set_trace()
error = []
for i, tree_di in enumerate(trees):
#for i, tree_di in enumerate([trees[-2]]):
    print('depth:', i+1)
    cnt = 0
    for i in range(attr.shape[0]):
        if tree_di.predict(attr[i]) != labels[i]:
            cnt += 1
    print('Train Error:', cnt/attr.shape[0])
    test_cnt = 0
    for i in range(test_attr.shape[0]):
        if tree_di.predict(test_attr[i]) != test_labels[i]:
            test_cnt += 1
    print('Test Error:', test_cnt/test_attr.shape[0])
    error.append([cnt/attr.shape[0], test_cnt/test_attr.shape[0]])

error = np.array(error)
print(np.mean(error, axis=0))
import pdb;pdb.set_trace()
