from abc import ABC, abstractmethod

import numpy as np


class IActivationFunction(ABC):

    @abstractmethod
    def calc(self, *args):
        pass

    @abstractmethod
    def derivative(self, *args):
        pass


class Relu(IActivationFunction):

    def calc(self, Z):
        return np.maximum(0, Z)

    def derivative(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def __repr__(self):
        return 'ReLu'


class Tanh(IActivationFunction):

    def calc(self, Z):
        return np.tanh(Z)

    def derivative(self, dA, Z):
        return dA * (1 - np.power(self.calc(Z), 2))

    def __repr__(self):
        return Tanh.__name__


class Sigmoid(IActivationFunction):

    def calc(self, Z):
        return 1./(1. + np.power(np.e, (-Z)))

    def derivative(self, dA, Z):
        A = self.calc(Z)
        return dA * (A * (1 - A))

    def __repr__(self):
        return Sigmoid.__name__


class Softmax(IActivationFunction):

    def calc(self, Z):
        return np.exp(Z) / np.sum(np.exp(Z), axis=0)

    def derivative(self, dA, Z):
        return - (dA - self.calc(Z))

    def __repr__(self):
        return Softmax.__name__
