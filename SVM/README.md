# Support Vector Machines (SVM) 

For running all svm algorithms, you will want to import the specific
svm version from the svm.py module. In this module, you have the
options of the primal and dual algorithms. An example of importing
the primal algorithm can be seen below. 

```python
from MLPy.svm.perceptron import PrimalSVM
```

Upon importing the class you will need to intialize a instance of this class.
To do this you will need to pass train data, in the orientation (NxD), test
data, in the orientation (NxD), a C value, and a gamma function that should only
take the argument of "t". Uppon initializes this class, you can run the
algorithm by calling the __call__ function of the instance and passing it an
integer value for T, or the number of epochs. All of this can be seen in the
following python cell.

```python
primal_svm = PrimalSVM(train_data, test_data, C, gamma_function)
primal_svm(T=100)
```

In order to run the dual algorithm, you will need to perform the following
import statement.


```python
from MLPy.svm.perceptron import DualSVM
```

Upon importing the class you will need to intialize a instance of this class.
To do this you will need to pass train data, in the orientation (NxD), test
data, in the orientation (NxD), a C value. Uppon initializes this class, you 
can run the algorithm by calling the __call__ function of the instance. All of 
this can be seen in the following python cell.

```python
dual_svm = DualSVM(train_data, test_data, C)
dual_svm()
```


