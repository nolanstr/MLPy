# Perceptron

For running all perceptron algorithms, you will want to improt the specific
perceptron version from the perceptron.py module. In this module, you have the
options of the Standard, Voted, and Average algorithms. An example of imprting
the Averaged algorithm can be seen below. 

'''python
from MLPy.Perceptron.perceptron import Standard
'''

After having imported this module, you will need to import data that can be
split into feature values and labels. The shape of feature values should be
(mxd) where m is the number of training examples and d is the dimensionality of
the data. Labels should be a one dimensional array of length d.

Once this data is loaded and ogranized, initialize a single instance of the
perceptron algorithm as is seen in the following line and run the perceptron by
calling the __call__ function with the desired number of epochs. 

'''python
perceptron = Standard(feature_values, labels)
perceptron(epochs)
'''

To evaluate the error on a test data set, organize the test data into feature
values and labels as was done previously and execute the following line.

'''python
error = perceptron.calc_error(feature_values, labels)
'''
