

class Fitness:
    '''
    Basic class that takes metric and returns idx for split.
    '''
    def __init__(self, metric):
        
        self.metric = metric

    def __call__(self, attributes, labels):
    
        return self.metric(attribues, labels)


