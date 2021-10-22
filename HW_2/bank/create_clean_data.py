import numpy as np

def _gen_clean_data(string, upd_idxs):
    train_list = []
    with open(string, 'r') as f:
        for line in f:
            line = line.split(',')
            line[-1] = line[-1][:-1]
            train_list.append(line)
    
    train = np.array(train_list)
    attr = train[:,:-1]
    labels = train[:,-1]
    
    for i in upd_idxs:
        column = attr[:,i].astype(float)
        med = np.median(column)
        above_idxs = np.where(column>med)[0]
        below_idxs = np.where(column<=med)[0]
        attr[above_idxs, i] = 1
        attr[below_idxs, i] = 0
    import pdb;pdb.set_trace()
    return attr, labels

train_list = []
with open('train.csv', 'r') as f:
    for line in f:
        line = line.split(',')
        line[-1] = line[-1][:-1]
        train_list.append(line)

train = np.array(train_list)
attr = train[:,:-1]

upd_idxs = []
for i in range(attr.shape[1]):

    try:
        float(attr[0,i])
        upd_idxs.append(i)
    except:
        continue


train_attr, train_labels = _gen_clean_data('train.csv', upd_idxs)
import pdb;pdb.set_trace()
np.save('clean_train_attr.npy', train_attr)
np.save('clean_train_labels.npy', train_labels)
test_attr, test_labels = _gen_clean_data('test.csv', upd_idxs)
np.save('clean_test_attr.npy', test_attr)
np.save('clean_test_labels.npy', test_labels)

import pdb;pdb.set_trace()
