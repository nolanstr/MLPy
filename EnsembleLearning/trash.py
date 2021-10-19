from copy import deepcopy

class A:
    def __init__(self, a):
        self._a= a


    def __call__(self):
        return [A(deepcopy(self)._a/2), A(deepcopy(self)._a/3)]
