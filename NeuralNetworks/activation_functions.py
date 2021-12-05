import numpy as np


def sigmoid(w, INPUT):
    
    '''
    input should have a 1 as the first value
    w shape should match input shape
    '''

    output = np.zeros(INPUT.shape)

    for i, layer in enumerate(w):
        
        output[i] = 1 / (1 + np.exp(-np.dot(layer, INPUT)))

    return output

def reverse_sigmoid(w, INPUT):
    
    '''
    input should have a 1 as the first value
    w shape should match input shape
    '''
        
    output = sigmoid(w, INPUT) * (1 - sigmoid(w, INPUT)) * INPUT 

    return output
