import numpy as np

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

train_attr, train_labels = _gen_clean_data('train.csv')
np.save('clean_train_attr.npy', train_attr)
np.save('clean_train_labels.npy', train_labels)
test_attr, test_labels = _gen_clean_data('test.csv')
np.save('clean_test_attr.npy', test_attr)
np.save('clean_test_labels.npy', test_labels)

import pdb;pdb.set_trace()
