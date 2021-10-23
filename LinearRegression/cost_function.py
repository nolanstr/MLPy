import numpy as np
from numba import jit

#@jit
def J(x, y, w):
    
    xw = np.matmul(w.T,x.T).reshape(y.shape)
    val = y - xw

    return 0.5 * np.sum(val**2)


#@jit
def DJ(x, y, w):
    djdw = np.zeros(w.shape)
    
    for j in range(w.shape[0]):
        
        xw = np.matmul(w.T,x.T).reshape(y.shape)

        val = y - xw
        nw = np.matmul(val.T, x[:,j].reshape(x[:,j].shape[0],1))[0]
        djdw[j] -= nw

    return djdw 
