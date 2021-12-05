import numpy as np
from .activation_functions import *

'''
Each layer consists of the activation function and the weights that go into it
such that the 
'''

forward_eval_dict = {'SIGMOID':sigmoid}
reverse_eval_dict = {'SIGMOID':reverse_sigmoid}

class HiddenLayer:

    def __init__(self, size, activation='sigmoid'):
        
        self.w = np.ones((size, size))
        self.forward_eval =  forward_eval_dict[activation.upper()]
        self.reverse_eval =  reverse_eval_dict[activation.upper()]

    def update_weights(self, w):

        self.w = w
    
    def forward_eval_layer(self, INPUT):
        '''
        Performs forward evaluation of a single layer.
        '''
        OUPUT = np.zeros(INPUT.shape)
        
        for i in range(INPUT.shape[0]):
            
            OUTPUT[i] = self.forward_eval(self.w[i], INPUT)
        
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

    def __init__(self, size, activation='sigmoid'):
        
        self.w = np.ones(size)
        self.forward_eval =  forward_eval_dict[activation.upper()]
        self.reverse_eval =  reverse_eval_dict[activation.upper()]

    def update_weights(self, w):

        self.w = w
    
    def forward_eval_layer(self, INPUT):
        '''
        Performs forward evaluation of a single layer.
        '''
        
        OUTPUT = np.dot(self.w, INPUT)
        
        return OUTPUT

    def reverse_eval_layer(self, INPUT, pred, true):
        '''
        Performs reverse evaluation (derivation) of a single layer
        This assumes a loss function of square loss
        '''

        dels =  (pred - true) * INPUT 
        
        return dels
