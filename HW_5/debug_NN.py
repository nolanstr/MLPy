import numpy as np
import sys
sys.path.append('../../')
sys.path.append('..\..')

from MLPy.NeuralNetworks.neural_network import NeuralNetwork

x = np.array([1,1,1]).reshape((1,3))
y = np.array([1])

gamma_0 = 1
d = 0.1
learning_rate = lambda t: gamma_0 / (1 + (gamma_0*t/d))

NN = NeuralNetwork(x, y, learning_rate, 2, 2)

weights = [np.array([[-1,-2,-3], [1,2,3]]), 
           np.array([[-1,-2,-3], [1,2,3]]),
           np.array([[-1,2,-1.5]])]

for i, w in enumerate(weights):

    NN.layers[i].set_weights(w)

import pdb;pdb.set_trace()
print(NN.forward_eval(x))


import pdb;pdb.set_trace()
