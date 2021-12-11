import numpy as np
import sys
sys.path.append('../..')
sys.path.append('..\..')
from MLPy.NeuralNetworks.layer import HiddenLayer, FinalLayer


class NeuralNetwork:

    def __init__(self, x, y, learning_rate, nodes, init_w='GAUSSIAN', 
                                                    hidden_layers=3):

        #self.x = np.hstack((np.ones((x.shape[0],1)),x))
        self.x = x
        self.y = y

        self.layers = [HiddenLayer(self.x.shape[1]+1, nodes, init_w)] + \
        [HiddenLayer(nodes, nodes, init_w) for _ in range(hidden_layers-1)] + \
                                [FinalLayer(self.x.shape[1]+1, nodes, init_w)]

        self.learning_rate = learning_rate
        
        self.idxs = np.arange(self.x.shape[0])
        
        self.loss = []

    def optimize(self, epochs=10):
        
        np.random.shuffle(self.idxs)

        for T in range(epochs):
            print(f'Epoch: {T+1}') 
            for true, x in zip(self.y[self.idxs], self.x[self.idxs]):

                pred = self.forward_eval(x)

                self.update_layer_weights(pred, true, self.learning_rate(T))
                self.loss.append(self.loss_function(true, pred))
        
        self.loss = np.array(self.loss)

    def compute_error(self, x, y):

        pred = np.sign(self.pred(x))
        pred[np.where(pred == 0)] = -1
        error = np.count_nonzero(pred - y) / y.shape[0]
        
        print(f'Error: {error}')

        return error

    def pred(self, x):

        pred = np.zeros(x.shape[0])

        for i in range(pred.shape[0]):
            pred[i] = self.forward_eval(x[i])

        return pred

    def loss_function(self, y, pred):

        return 0.5 * ((pred - y)**2)

    def update_layer_weights(self, pred, true, gamma):
        '''
        Work from here, wee need to find a way to take all of the partials and
        collapse them into values that make sense, this should be done in the
        reversed order of the layers list (i.e final layer first).

        This will also need to compute the partials here by calling self.reverse
       _eval
        '''
        self.reverse_eval(pred, true, gamma)

    def forward_eval(self, x):
        
        x = np.concatenate((np.ones(1), x))
        for layer in self.layers:
    
            x = layer.forward_eval_layer(x)
        
        return x


    def reverse_eval(self, pred, true, gamma):
        
        for i, layer in enumerate(reversed(self.layers)):
        
            if i == 0:
                layer.reverse_eval_layer(pred, true, gamma)
            else:
                layer.reverse_eval_layer(prev_layer, gamma)
            prev_layer = layer

        return None #ders
            
    

