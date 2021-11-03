import numpy as np
import sys
sys.path.append('../../')
sys.path.append('..\..')

from MLPy.Perceptron.perceptron import Voted

data = np.genfromtxt('bank-note/train.csv', delimiter=',')

perceptron = Voted(data[:,0:-1], data[:,-1])
perceptron(10)

test_data = np.genfromtxt('bank-note/test.csv', delimiter=',')

error = perceptron.calc_error(test_data[:,0:-1], test_data[:,-1])
print('Average Test error:', error)
print('Weight Vector: All values stored in w_values_problem_2b.txt')
print('Counts Vector: All values stored in c_values_problem_2b.txt')
np.savetxt('w_values_problem_2b.txt', np.array(perceptron.w).reshape((260,5)))
np.savetxt('c_values_problem_2b.txt', np.array(perceptron.c))

import pdb;pdb.set_trace()

