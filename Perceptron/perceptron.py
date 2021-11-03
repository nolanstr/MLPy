import numpy as np

class Standard:

    def __init__(self, fv, labels):
        """
        self.fv -- feature values
        self.labels -- labels
        """

        self.fv = np.hstack((np.ones((fv.shape[0],1)), fv))
        self.labels = labels
        self.labels[np.where(labels == 0)] = -1
        self.idxs = np.arange(self.fv.shape[0])

    def __call__(self, T, r=0.5):

        np.random.shuffle(self.idxs)

        self.w = np.zeros(self.fv.shape[1]).reshape((1,-1))
        
        for i in range(T):
            
            for xi, yi in zip(self.fv, self.labels):

                pred = np.sign(np.matmul(self.w, xi)[0])

                if pred != yi:
                    self.w += (r * xi * yi)
                    
    def calc_error(self, fv, labels):
        
        fv = np.hstack((np.ones((fv.shape[0],1)), fv))
        labels = labels 
        labels[np.where(labels == 0)] = -1

        pred = np.sign(np.matmul(fv, self.w.T).flatten())
        import pdb;pdb.set_trace()
        inc_pred = np.where(pred-labels != 0)[0].shape[0]

        return inc_pred / labels.shape[0]

class Voted:

    def __init__(self, fv, labels):
        """
        self.fv -- feature values
        self.labels -- labels
        """

        self.fv = np.hstack((np.ones((fv.shape[0],1)), fv))
        self.labels = labels
        self.labels[self.labels==0] = -1

        self.idxs = np.arange(self.fv.shape[0])

    def __call__(self, T, r=0.5):

        np.random.shuffle(self.idxs)

        self.w = [np.zeros(self.fv.shape[1]).reshape((1,-1))]
        self.m = [np.zeros(self.fv.shape[1]).reshape((1,-1))]
        self.c = [0]

        for i in range(T):
            
            for xi, yi in zip(self.fv, self.labels):

                pred = np.sign(np.matmul(self.w[-1], xi)[0])

                if pred != yi:
                    
                    self.w.append(self.w[-1] + (r * xi * yi))
                    self.c.append(1)

                else:
                    self.c[-1] += 1
        
        self.c = np.array(self.c)

        import pdb;pdb.set_trace()
    
    def calc_error(self, fv, labels):

        fv = np.hstack((np.ones((fv.shape[0],1)), fv))
        labels = labels 
        labels[np.where(labels == 0)] = -1

        pred = np.array([np.sign(np.matmul(fv,
            self.w[i].T).flatten()) for i in range(self.c.shape[0])])
        pred[np.where(pred == 0)] = -1

        for row in range(pred.shape[0]):
            pred[row] *= self.c[row]
        
        voted_pred = np.sign(np.sum(pred, axis=0))
        voted_pred[np.where(voted_pred == 0)] = -1
        inc_pred = np.where(voted_pred - labels != 0)[0].shape[0] 
        
        return inc_pred / labels.shape[0]

class Averaged:

    def __init__(self, fv, labels):
        """
        self.fv -- feature values
        self.labels -- labels
        """

        self.fv = np.hstack((np.ones((fv.shape[0],1)), fv))
        self.labels = labels
        self.labels[self.labels==0] = -1

        self.idxs = np.arange(self.fv.shape[0])

    def __call__(self, T, r=0.5):

        np.random.shuffle(self.idxs)

        self.w = [np.zeros(self.fv.shape[1]).reshape((1,-1))]
        self.m = [np.zeros(self.fv.shape[1]).reshape((1,-1))]
        self.c = [0]

        for i in range(T):
            
            for xi, yi in zip(self.fv, self.labels):

                pred = np.sign(np.matmul(self.w[-1], xi)[0])

                if pred != yi:
                    
                    self.w.append(self.w[-1] + (r * xi * yi))
                    self.c.append(1)

                else:
                    self.c[-1] += 1
        
        self.c = np.array(self.c)

        import pdb;pdb.set_trace()
    
    def calc_error(self, fv, labels):

        fv = np.hstack((np.ones((fv.shape[0],1)), fv))
        labels = labels 
        labels[np.where(labels == 0)] = -1

        pred = np.array([np.matmul(fv,
            self.w[i].T).flatten() for i in range(self.c.shape[0])])

        for row in range(pred.shape[0]):
            pred[row] *= self.c[row]
        
        voted_pred = np.sign(np.sum(pred, axis=0))
        voted_pred[np.where(voted_pred == 0)] = -1
        inc_pred = np.where(voted_pred - labels != 0)[0].shape[0] 
        
        return inc_pred / labels.shape[0]

