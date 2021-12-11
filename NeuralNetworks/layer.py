import numpy as np
from .activation_functions import *

'''
Each layer consists of the activation function and the weights that go into it
such that the 
'''

forward_eval_dict = {'SIGMOID':sigmoid}
reverse_eval_dict = {'SIGMOID':reverse_sigmoid}

class HiddenLayer:

    def __init__(self, input_size, nodes, init_w, activation='sigmoid'):

        if init_w.upper() == 'ZEROS':
            self.w = np.zeros((nodes-1, input_size))
        elif init_w.upper() == 'ONES':
            self.w = np.ones((nodes-1, input_size))
        elif init_w.upper() == 'GAUSSIAN':
            self.w = np.random.normal(size=(nodes-1, input_size))
        
        self.forward_eval =  forward_eval_dict[activation.upper()]
        self.reverse_eval =  reverse_eval_dict[activation.upper()]

    def update_weights(self, gamma):
        
        self.w = self.w - self.w_dels * gamma 

    def set_weights(self, w):

        self.w = w
    
    def forward_eval_layer(self, INPUT):
        '''
        Performs forward evaluation of a single layer.
        '''

        self.Z = INPUT
        OUTPUT = np.ones(self.w.shape[0]+1)

        for i in range(self.w.shape[0]):
            
            try:
                OUTPUT[i+1] = self.forward_eval(self.w[i], INPUT)
            except:
                pass

        return OUTPUT

    def reverse_eval_layer(self, prev_layer, gamma):
        '''
        Performs reverse evaluation (derivation) of a single layer by taking the
        previous layer 
        '''
        
        self.w_dels = np.zeros(self.w.shape)
        self.layer_dels = np.zeros(self.w.shape[1])

        for i in range(self.w.shape[0]):
            for j in range(self.w.shape[1]):
                self.w_dels[i,j] = prev_layer.layer_dels[i+1] * \
                        (prev_layer.Z[i+1] * (1-prev_layer.Z[i+1])) * self.Z[j]

                self.layer_dels[j] += prev_layer.layer_dels[i+1] * self.w[i,j] \
                                * prev_layer.Z[i+1] * (1 - prev_layer.Z[i+1])
        
        self.update_weights(gamma)

        return None

class FinalLayer:

    def __init__(self, input_size, nodes, init_w):
        
        if init_w.upper() == 'ZEROS':
            self.w = np.zeros((1, nodes))
        elif init_w.upper() == 'ONES':
            self.w = np.ones((1, nodes))
        elif init_w.upper() == 'GAUSSIAN':
            self.w = np.random.normal(size=(1, nodes))
        

    def update_weights(self, gamma):

        try:
            self.w = self.w - self.w_dels * gamma 
        except:
            import pdb;pdb.set_trace()

    def set_weights(self, w):

        self.w = w
    
    def forward_eval_layer(self, INPUT):
        '''
        Performs forward evaluation of a single layer.
        '''
        OUTPUT = np.dot(self.w, INPUT)
        self.Z = INPUT 

        return OUTPUT[0]

    def reverse_eval_layer(self, pred, true, gamma):
        '''
        Performs reverse evaluation (derivation) of a single layer
        '''

        self.w_dels = (pred - true) * self.Z
        self.layer_dels = ((pred-true) * self.w).flatten()

        self.update_weights(gamma)
        
        return None
