import numpy as np

def _gen_clean_data(string):
    train_list = []
    with open(string+'.csv', 'r') as f:
        for line in f:
            line = line.split(',')
            line[-1] = line[-1][:-1]
            train_list.append(line)
    
    train = np.array(train_list).astype(float)
    np.save(string+'.npy', train)

_gen_clean_data('train')
_gen_clean_data('test')
import pdb;pdb.set_trace()

import pdb;pdb.set_trace()
