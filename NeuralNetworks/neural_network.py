import numpy as np

from .layer import HiddenLayer, FirstLayer, FinalLayer


def NeuralNetwork:

    def __init__(self, x, y, learning_rate, hidden_layers=3):

        self.layers = [HiddenLayer(x.shape[0]) for _ in range(hidden_layers)] +\
                            [FinalLayer(x.shape[0])]
    
        self.learning_rate = learning_rate
        self.x_data = x
        self.y = y
        
        self.idxs = np.arange(self.x_data.shape[0])
        
    def optimize(self, epochs=100):
        
        np.random.shuffle(self.idxs)

        for T in epochs:
            
            for true, x in zip(self.y[idxs], self.x_data[idxs])

                pred = self.forward_eval(x)

                if pred != true:
                   self.update_layer_weights(pred, true)

                else:
                    pass
    
    def update_layer_weights(self, pred, true):
        '''
        Work from here, wee need to find a way to take all of the partials and
        collapse them into values that make sense, this should be done in the
        reversed order of the layers list (i.e final layer first).

        This will also need to compute the partials here by calling self.reverse
       _eval
        '''

    def forward_eval(self, x):
        
        for layer in self.layers:

            x = layer.forward_eval_layer(x)
        
        return x


    def reverse_eval(self, x, pred, true):
        
        ders = []
        INPUT = x

        for i, layer in enumerate(reversed(self.layers)):
            
            if i == 1:
                INPUT = layer.reverse_eval(INPUT)
                ders.append(INPUT)

            INPUT = layer.reverse_eval(INPUT)
            ders.append(INPUT)
        
        return ders
            
    

