import numpy as np
from .activation_functions import *

'''
Each layer consists of the activation function and the weights that go into it
such that the 
'''

forward_eval_dict = {'SIGMOID':sigmoid}
reverse_eval_dict = {'SIGMOID':reverse_sigmoid}

class HiddenLayer:

    def __init__(self, input_size, nodes, activation='sigmoid'):
        
        self.w = np.ones((nodes, input_size))
        self.forward_eval =  forward_eval_dict[activation.upper()]
        self.reverse_eval =  reverse_eval_dict[activation.upper()]

    def update_weights(self, w):

        self.w = w

    def set_weights(self, w):

        self.w = w
    
    def forward_eval_layer(self, INPUT):
        '''
        Performs forward evaluation of a single layer.
        '''
        OUTPUT = np.ones(self.w.shape[1])
        
        for i in range(self.w.shape[0]):

            OUTPUT[i+1] = self.forward_eval(self.w[i], INPUT)
        print(OUTPUT)
        return OUTPUT

    def reverse_eval_layer(self, INPUT):
        '''
        Performs reverse evaluation (derivation) of a single layer
        '''
        dels = np.zeros((INPUT.shape, INPUT.shape))

        for i in range(1, dels.shape[0]):

            dels[i] = self.reverse_eval(self.w[i], INPUT)
        
        return dels

class FinalLayer:

    def __init__(self, input_size, nodes):
        
        self.w = np.ones((1, input_size))

    def update_weights(self, w):

        self.w = w

    def set_weights(self, w):

        self.w = w
    
    def forward_eval_layer(self, INPUT):
        '''
        Performs forward evaluation of a single layer.
        '''
        import pdb;pdb.set_trace()            
        OUTPUT = np.dot(self.w, INPUT)
        
        return np.sign(OUTPUT[0])

    def reverse_eval_layer(self, INPUT, pred, true):
        '''
        Performs reverse evaluation (derivation) of a single layer
        '''

        dels = (pred - true) * INPUT 
        
        return dels

