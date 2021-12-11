# Neural Network (NN)

In order to run this NN package you will need to import
the NN class. This is done in the following section of code.

```python
from MLPy.NeuralNetworks.neural_network import NeuralNetworl
```

Upon importing this class, you will need to load in your input train data in the
shape (MxD) where M is the number of instances and D is the dimensionality. You
will also need to import your output data in the shape (M,). This is a flat
array that is one dimensional. Furthermore you will need to initialize a
learning rate function (gamma) that takes the current epoch number as an
argument. Finally, you will need to specify the width of each hidden layer (w). 


```python
NN = NeuralNetwork(x, y, gamma, w)
NN.optimize(epochs=100)
```
Finally, to check the train and test error, you can run the following line of
code where x_test is your test input data and y_test is your test output data.


```python
train_error = NN.compute_error(x_train, y_train)
test_error = NN.compute_error(x_test, y_test)
```
!!
Please run code with GitBash!
!!

