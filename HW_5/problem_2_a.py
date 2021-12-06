import numpy as np
import sys
sys.path.append('../../')
sys.path.append('..\..')

from MLPy.NeuralNetworks.neural_network import NeuralNetwork

train_data = np.genfromtxt('bank-note/train.csv', delimiter=',')
test_data = np.genfromtxt('bank-note/test.csv', delimiter=',')

x_train = train_data[:,:-1]
y_train = train_data[:,-1]

x_test = test_data[:,:-1]
y_test = test_data[:,-1]

gamma_0 = 1
d = 0.1
learning_rate = lambda t: gamma_0 / (1 + (gamma_0*t/d))

NN = NeuralNetwork(x_train, y_train, learning_rate, 5)
import pdb;pdb.set_trace()
