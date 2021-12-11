import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
sys.path.append('..\..')

from MLPy.NeuralNetworks.neural_network import NeuralNetwork

train_data = np.genfromtxt('bank-note/train.csv', delimiter=',')
test_data = np.genfromtxt('bank-note/test.csv', delimiter=',')

x_train = train_data[:,:-1]
y_train = train_data[:,-1]
y_train[np.where(y_train == 0)] = -1

x_test = test_data[:,:-1]
y_test = test_data[:,-1]
y_test[np.where(y_test == 0)] = -1

    
def test_width(gamma_0, d, w):
    learning_rate = lambda t: gamma_0 / (1 + (gamma_0*t/d))

    NN = NeuralNetwork(x_train, y_train, learning_rate, w)

    NN.optimize()

    print(f'Width: {w}')
    print('Train')
    error = NN.compute_error(x_train, y_train)
    print('Test')
    test_error = NN.compute_error(x_test, y_test)

    plt.plot(np.arange(NN.loss.shape[0]), NN.loss)
    plt.xlabel('Updates')
    plt.ylabel('Loss')
    plt.title('Loss vs. Neural Network updates')
    plt.grid()

    plt.show()
    
    plt.clf()

ws = [5,10,25,50,100]

gs = [7e-2]*len(ws)
ds = [7e-2]*len(ws)

for gamma_0, d, w in zip(gs, ds, ws):
    
    test_width(gamma_0, d, w)

