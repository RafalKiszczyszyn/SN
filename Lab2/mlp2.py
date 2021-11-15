from __future__ import annotations
from dataclasses import dataclass
from typing import List

import numpy as np
import pickle

from Lab2.mltoolkit import IActivationFunction, Stats, DataSet, accuracy


@dataclass
class Layer:
    inputs: int
    outputs: int
    activation: IActivationFunction


@dataclass
class Params:
    W: dict
    b: dict

    def copy(self):
        W_copy = {}
        b_copy = {}
        for layer_id in self.W:
            W_copy[layer_id] = np.array(self.W[layer_id], copy=True)
            b_copy[layer_id] = np.array(self.b[layer_id], copy=True)
        return Params(W=W_copy, b=b_copy)


@dataclass
class ParamsGrads:
    dW: dict
    db: dict


@dataclass
class CalcCache:
    A: dict
    Z: dict


@dataclass
class BestParams:
    params: Params
    cost_dev: float
    epoch: int


class MultilayerPerceptron:

    def __init__(self, layers: List[Layer], mean: float, sigma: float):
        self.mean = mean
        self.sigma = sigma
        self._layers = layers

        self._params = self._generate_params()

    def _generate_params(self):
        params = Params(W={}, b={})
        for layer_id, layer in enumerate(self._layers, start=1):
            params.W[layer_id] = np.random.normal(self.mean, self.sigma, (layer.outputs, layer.inputs))
            params.b[layer_id] = np.random.normal(self.mean, self.sigma, (layer.outputs, 1))

        return params

    def _forward_propagation(self, X):
        cache = CalcCache(A={}, Z={})

        A_curr = X
        for layer_id, layer in enumerate(self._layers, start=1):
            A_prev = A_curr

            activation = layer.activation
            W_curr = self._params.W[layer_id]
            b_curr = self._params.b[layer_id]
            A_curr, Z_curr = self._forward_propagation_step(
                A_prev=A_prev, W_curr=W_curr, b_curr=b_curr, activation=activation)

            cache.A[layer_id - 1] = A_prev
            cache.Z[layer_id] = Z_curr

        cache.A[len(self._layers)] = A_curr
        return A_curr, cache

    @staticmethod
    def _forward_propagation_step(A_prev, W_curr, b_curr, activation: IActivationFunction):
        Z = np.dot(W_curr, A_prev) + b_curr
        return activation.calc(Z), Z

    def _backward_propagation(self, Y, cache: CalcCache):
        grads = ParamsGrads(dW={}, db={})

        dA_prev = Y
        for layer_id, layer in reversed(list(enumerate(self._layers, start=1))):
            activation = layer.activation

            dA_curr = dA_prev

            A_prev = cache.A[layer_id - 1]
            Z_curr = cache.Z[layer_id]
            W_curr = self._params.W[layer_id]

            dA_prev, dW_curr, db_curr = self._backward_propagation_step(
                dA_curr=dA_curr, W_curr=W_curr, Z_curr=Z_curr, A_prev=A_prev, activation=activation)

            grads.dW[layer_id] = dW_curr
            grads.db[layer_id] = db_curr

        return grads

    @staticmethod
    def _backward_propagation_step(dA_curr, W_curr, Z_curr, A_prev, activation: IActivationFunction):
        m = A_prev.shape[1]

        dZ_curr = activation.derivative(dA_curr, Z_curr)
        dW_curr = np.dot(dZ_curr, A_prev.T) / m
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr

    def _update(self, grads: ParamsGrads, eta: float):
        for layer_id, layer in enumerate(self._layers, start=1):
            self._params.W[layer_id] -= eta * grads.dW[layer_id]
            self._params.b[layer_id] -= eta * grads.db[layer_id]

    def _cost(self, Y, Y_hat):
        return (- np.log(Y_hat) * Y).sum() / len(Y)

    def fit2(self,
             train_data_set: DataSet,
             dev_data_set: DataSet,
             epochs: int,
             eta: float,
             batch_size: int,
             max_train_accuracy: float,
             max_cost_dev_rise: float):

        best = BestParams(
            params=self._params.copy(),
            cost_dev=np.finfo('d').max,
            epoch=-1)
        stats = Stats()

        batches = int(len(train_data_set.labels) / batch_size)
        batches = batches if batches > 0 else 1

        for epoch in range(epochs):
            permutation = np.random.permutation(len(train_data_set.labels))
            for batch in np.split(permutation, batches):
                X_batch = train_data_set.X[batch].T
                Y_batch = train_data_set.Y[batch].T
                Y_hat, cache = self._forward_propagation(X_batch)

                grads = self._backward_propagation(Y=Y_batch, cache=cache)
                self._update(grads=grads, eta=eta)

            Y_train_hat = self.predict(X=train_data_set.X)
            cost_train = self._cost(Y=train_data_set.Y, Y_hat=Y_train_hat)
            stats.cost_train_history.append(cost_train)

            Y_dev_hat = self.predict(X=dev_data_set.X)
            cost_dev = self._cost(Y=dev_data_set.Y, Y_hat=Y_dev_hat)
            stats.cost_dev_history.append(cost_dev)

            if cost_dev < best.cost_dev:
                best = BestParams(
                    params=self._params.copy(),
                    cost_dev=cost_dev,
                    epoch=epoch)
            elif cost_dev >= (1 + max_cost_dev_rise) * best.cost_dev:
                self._params = best.params.copy()
                return best, stats, 'COST'

            if accuracy(train_data_set.labels, Y_train_hat) > max_train_accuracy:
                self._params = best.params.copy()
                return best, stats, 'ACCURACY'

        self._params = best.params.copy()
        return best, stats, 'EPOCHS'

    def predict(self, X):
        Y_hat, _ = self._forward_propagation(X.T)
        return Y_hat.T


def serialize_model(mlp: MultilayerPerceptron, filename):
    with open(filename, 'wb') as f:
        pickle.dump(mlp, f)


def deserialize_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
