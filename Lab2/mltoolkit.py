from abc import ABC, abstractmethod
from typing import List

import numpy as np

from dataclasses import dataclass, field


@dataclass
class DataSet:
    X: np.ndarray
    Y: np.ndarray
    labels: np.ndarray


@dataclass
class Stats:
    cost_train_history: List[float] = field(default_factory=lambda: [])
    cost_dev_history: List[float] = field(default_factory=lambda: [])


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
        return 'TanH'


class Softmax(IActivationFunction):

    def calc(self, Z):
        return np.exp(Z) / np.sum(np.exp(Z), axis=0)

    def derivative(self, dA, Z):
        return - (dA - self.calc(Z))

    def __repr__(self):
        return 'Softmax'


def accuracy(labels: np.ndarray, Y_hat: np.ndarray):
    predicted_labels = Y_hat.argmax(axis=1)
    matches = predicted_labels == labels
    correct = matches.sum()
    return correct / len(labels)
