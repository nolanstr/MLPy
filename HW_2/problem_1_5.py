import numpy as np

x = np.array([[1,-1,2],[1,1,3],[-1,1,0],[1,2,-4],[3,-1,-1]])
y = np.array([1,4,-1,-2,0]).reshape((5,1))
w = np.array([-1,1,-1]).reshape((3,1))
b = -1

djdw = np.zeros((3,1))

for i in range(x.shape[0]):

    x_i = x[i].reshape((x[i].shape[0],1))
    y_i = y[i]

    val = y_i - np.matmul(w.T, x_i) - b
    djdw += val[0] * -x_i 

djdb = 0

for i in range(x.shape[0]):

    x_i = x[i].reshape((x[i].shape[0],1))
    y_i = y[i]

    val = y_i - np.matmul(w.T, x_i) - b
    djdb += -val[0] 

print(djdb)


print('optimal solution, part c')
x = np.hstack((np.ones((x.shape[0],1)), x))

w_star = np.matmul(np.linalg.inv(np.matmul(x.T,x)), np.matmul(x.T,y))

print(w_star)


print('Stochastic Gradient descent, part d')

def djdw(x, y, w):
    
    djdw = np.zeros((4,1))

    for i in range(x.shape[0]):

        x_i = x[i].reshape((x[i].shape[0],1))
        y_i = y[i]
        val  = y_i - np.matmul(w.T, x_i)
        djdw += val[0] * -x_i

    return djdw


w_0 = np.zeros((4,1))
#b is the first values in w_0
r = 0.1
print(w_0)
for i in range(5):
    
    y_i = y[i]
    x_i = x[i].reshape((4,1))


    djdw_i = djdw(x, y, w_0)
    #print(djdw_i)
    for j in range(4):
        w_0[j] += r * (y_i - np.matmul(w_0.T, x_i))[0] * x[i,j]
        #import pdb;pdb.set_trace()
    print(w_0)

    #import pdb;pdb.set_trace()





