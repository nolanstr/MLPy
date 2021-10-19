from decision_tree import DecisionTree

class AdaBoost:

    def __init__(self, data, fitness, depth, T):

        D = np.ones((data.shape[0],1)) / data.shape[0]
        self.attr = data[0]
        self.labels = data[1]
        self.depth = depth
        self.T = T

    #def call(self):
        


