import numpy as np
import sys
sys.path.append('../../')
sys.path.append('..\..')

from MLPy.Perceptron.perceptron import Standard

data = np.genfromtxt('bank-note/train.csv', delimiter=',')

perceptron = Standard(data[:,0:-1], data[:,-1])
perceptron(10)

test_data = np.genfromtxt('bank-note/test.csv', delimiter=',')

error = perceptron.calc_error(test_data[:,0:-1], test_data[:,-1])
print('Average Test error:', error)
print('Weight Vector:', perceptron.w)

