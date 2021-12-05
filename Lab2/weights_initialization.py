from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

from Lab2 import models


class IWeightsInitializationStrategy(ABC):

    @abstractmethod
    def initialize(self, layer: models.Layer) -> Tuple[np.ndarray, np.ndarray]:
        pass


class RandomWeightsInitializationStrategy(IWeightsInitializationStrategy):

    def __init__(self, mean: float = 0., sigma: float = 0.01):
        self.mean = mean
        self.sigma = sigma

    def initialize(self, layer: models.Layer) -> Tuple[np.ndarray, np.ndarray]:
        W = np.random.normal(self.mean, self.sigma, (layer.outputs, layer.inputs))
        b = np.zeros((layer.outputs, 1))
        return W, b


class XavierWeightsInitializationStrategy(IWeightsInitializationStrategy):

    def __init__(self, mean: float = 0.):
        self.mean = mean

    def initialize(self, layer: models.Layer) -> Tuple[np.ndarray, np.ndarray]:
        W = np.random.normal(
            self.mean,
            np.sqrt(2 / (layer.outputs + layer.inputs)),
            (layer.outputs, layer.inputs))
        b = np.zeros((layer.outputs, 1))
        return W, b


class HeWeightsInitializationStrategy(IWeightsInitializationStrategy):

    def __init__(self, mean: float = 0.):
        self.mean = mean

    def initialize(self, layer: models.Layer) -> Tuple[np.ndarray, np.ndarray]:
        W = np.random.normal(self.mean, np.sqrt(2 / layer.inputs), (layer.outputs, layer.inputs))
        b = np.zeros((layer.outputs, 1))
        return W, b
