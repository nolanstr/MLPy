# Perceptron

To import and run this ensemble learning code, you must import the specific
class you wish to use. The classes are placed in modules that are named after
the ensemble learning method that they perform. The available ensemble learning
algorithms are Adaboost, Bagging, and Random Forrest. An example of loading and
runnning the Bagging code can be seen below.


First you will need to import the specific ensemble learning class as well as
the chosen fitness metric that will be used to evaluate and split your trees.

```python
from MLPy.EnsembleLearning.bagging import Bagging
from MLPy.EnsembleLearning.fitness_metric.entropy import Entropy
```

Once this is completed, you will need to load in your train and test data as
numpy arrays. The train and test data will also need to be divided into an
attributes array and a labels array. The shape of the attributes array is (mxd),
where m represent the total number of data points and d represents the
dimensionality of the data set. The labels will be a one dimensional numpy array
of length m.

Upon loading and organizing this data, a single of instance of the Bagging class
can be created and the algorithm can be ran by additionally specifying the
number of trees, T, to be bagged. This argument may change slightly for other
algorithms as they have slightly different needs. By executing the following
code, an instance of both the fitness metric and the Bagging class will be
creared and the algorithm will begin running. 

```python
T  = number of trees
fitness = Entropy()
bag = Bagging((train_attributes, train_labels), fitness, T)
```

To evaluate the error on either datasets, you will need to run the followin
gline of code and pass your specified data to it in the same orientation as was
stated previously.

```python
error_train = bag.find_all_bagging_errrors()
```

or

```python
error_test = bag.find_test_bagging_errrors((test_attr, test_labels))
```
