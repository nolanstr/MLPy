import numpy as np
import numba

@jit
def J(x, y, w):

    return 0.5 * np.sum((y - np.matmul(w.T, x))**2)


@jit
def DJ(x, y, w):
    djdw = np.zeros(w.shape)

    for i in range(w.shape[0])
        x_i = x[i].reshape((x[i].shape, 1))
        y_i = y[i]
        val = y_i = np.matmul(w.T, x_i)
        djdw += val[0] * -x_i

    return djdw 
