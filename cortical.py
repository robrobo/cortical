import numpy as np

ACTIVATIONTHRESHOLD = 0.4
CONNECTIONADJUSTMENT = 0.01
HISTORYLENGTH = 10
RETRACTIONTHRESHOLD = 0.5
RETRACTIONADJUSTMENT = 0.01
STARVATIONTHRESHOLD = 0.1
STARVATIONADJUSTMENT = 0.01
WEAKEN = 0.99


class CortialNN():
    def __init__(self,shape):
        self.shape=shape
        self.connections = np.random.rand(shape,shape)/shape
        np.fill_diagonal(self.connections, 0)
        self.gradient = np.zeros((shape,shape))
        self.active = np.int_(np.zeros(shape))
        self.previous = np.int_(np.zeros(shape))
        self.history = np.zeros((HISTORYLENGTH,shape))

    def tick(self,signal=None):
        self.history = np.roll(self.history, 1, axis=0)
        self.gradient = np.zeros((self.shape, self.shape))
        current = np.copy(self.previous)
        if signal != None:
            current[signal == 1] = 1
        temp = np.sum(current * self.connections, axis=-1)
        self.active[temp < ACTIVATIONTHRESHOLD] = 0
        self.active[temp >= ACTIVATIONTHRESHOLD] = 1
        if signal != None:
            self.active[signal == 1] = 1

        self.gradient += (self.previous & self.active[:, None])*CONNECTIONADJUSTMENT
        self.gradient -= (self.active & self.active[:,None])*CONNECTIONADJUSTMENT
        self.history[0] = np.copy(self.previous)
        self.gradient -= (np.sum(self.history, axis=0) / HISTORYLENGTH >= RETRACTIONTHRESHOLD) * RETRACTIONADJUSTMENT
        self.gradient += (np.sum(self.history, axis=0) / HISTORYLENGTH < STARVATIONTHRESHOLD) * STARVATIONADJUSTMENT
        np.fill_diagonal(self.gradient, 0)

        self.previous = np.copy(self.active)
        self.connections += self.gradient
        self.connections = self.connections * WEAKEN

