# Logistic Regression

In order to run this logistic regression (LR) package you will need to import
the LR class. This is done in the following section of code.

```python
from MLPy.LogisticRegression.logistic_regression import LogisticRegression
```

Upon importing this class, you will need to load in your input train data in the
shape (MxD) where M is the number of instances and D is the dimensionality. You
will also need to import your output data in the shape (M,). This is a flat
array that is one dimensional. Furthermore you will need to initialize a
learning rate function (gamma) that takes the current epoch number as an
argument. Finally, there are two algorthms you can choose from. The first is the
marginal likelihood (ML) algorithm, an LR object is created for this algorithm
and ran with the following lines of code.


```python
LR = LogisticRegression(x, y, gamma, method='MLE')
LR(T=100)
```

If you would rather use the maximum a priori estimate that can be done with the
following lines of code. Note, this assumes a standard Gaussian prior with the
given standard deviation (std).


```python
LR = LogisticRegression(x, y, gamma, std=1, method='MAP')
LR(T=100)
```

Finally, to check the train and test error, you can run the following line of
code where x_test is your test input data and y_test is your test output data.


```python
LR.get_test_train_error(x_test, y_test)
```

