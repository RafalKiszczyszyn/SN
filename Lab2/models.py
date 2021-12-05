from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass, field

import numpy as np

from . import activations


@dataclass
class DataSet:
    X: np.ndarray
    Y: np.ndarray
    labels: np.ndarray


@dataclass
class Layer:
    inputs: int
    outputs: int
    activation: activations.IActivationFunction


@dataclass
class Parameters:
    W: Dict[int, np.ndarray]
    b: Dict[int, np.ndarray]


@dataclass
class Gradients:
    dW: Dict[int, np.ndarray]
    db: Dict[int, np.ndarray]


def from_layers(layers: List[Layer], fillFunc: Callable[[Tuple], np.ndarray]) \
        -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    W, b = {}, {}
    for layer_id, layer in enumerate(layers, start=1):
        W[layer_id] = fillFunc((layer.outputs, layer.inputs))
        b[layer_id] = fillFunc((layer.outputs, 1))
    return W, b


def copy(W: Dict[int, np.ndarray], b: Dict[int, np.ndarray]) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    W_copy, b_copy = {}, {}
    for layer_id in W:
        W_copy[layer_id] = np.array(W[layer_id], copy=True)
        b_copy[layer_id] = np.array(b[layer_id], copy=True)
    return W_copy, b_copy


@dataclass
class CalculationsCache:
    A: dict
    Z: dict


@dataclass
class CostHistory:
    cost_train: List[float] = field(default_factory=lambda: [])
    cost_dev: List[float] = field(default_factory=lambda: [])


@dataclass
class AccuracyHistory:
    accuracy_train: List[float] = field(default_factory=lambda: [])
    accuracy_dev: List[float] = field(default_factory=lambda: [])


@dataclass
class Statistics:
    cost_history: CostHistory = field(default_factory=lambda: CostHistory(cost_train=[], cost_dev=[]))
    accuracy_history: AccuracyHistory = field(default_factory=lambda: AccuracyHistory(accuracy_train=[], accuracy_dev=[]))


@dataclass
class BestParameters:
    epoch: int
    params: Parameters
    cost_train: float
    accuracy_train: float
    cost_dev: float
    accuracy_dev: float
