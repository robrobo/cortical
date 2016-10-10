import numpy as np

class InputGenerator():
    def __init__(self, shape, signals, rates = 1):
        self.shape = shape
        self.signals = signals
        self.rates = rates
        self.binRates = np.cumsum(rates/np.sum(rates))

    def getInput(self, count=1):
        return self.getSignal(count=count)

    def getSignal(self, count=1, ind=None):
        if ind == None:
            return self.signals[np.digitize(np.random.rand(count),self.binRates)]
        else:
            return self.signals[ind]