from __future__ import annotations

from typing import List, Union

import numpy as np
import pickle

from Lab2 import models, activations, optimizers, toolkit
from Lab2.weights_initialization import IWeightsInitializationStrategy, RandomWeightsInitializationStrategy, \
    XavierWeightsInitializationStrategy, HeWeightsInitializationStrategy


# --- Exceptions ---

class EarlyStopping(Exception):

    ACCURACY = 'Accuracy'
    OVERFITTING = 'Overfitting'
    EPOCHS = 'Epochs'

    def __init__(self, reason):
        super().__init__(f"Early stopping for reason='{reason}'")
        self._reason = reason

    @staticmethod
    def overfitting():
        return EarlyStopping(reason=EarlyStopping.OVERFITTING)

    @staticmethod
    def accuracy():
        return EarlyStopping(reason=EarlyStopping.ACCURACY)

    @staticmethod
    def epochs():
        return EarlyStopping(reason=EarlyStopping.EPOCHS)

    @property
    def reason(self):
        return self._reason


# --- Implementations ---


class MultilayerPerceptron:

    def __init__(self, layers: List[models.Layer],
                 weights_initialization_strategy: Union[IWeightsInitializationStrategy, None] = None):
        self.layers = layers
        self.weights_initialization_strategy = weights_initialization_strategy \
            if weights_initialization_strategy \
            else RandomWeightsInitializationStrategy(mean=0, sigma=0.01)

        self.parameters = self._init_weights()

    def _init_weights(self):
        params = models.Parameters(W={}, b={})
        for layer_id, layer in enumerate(self.layers, start=1):
            params.W[layer_id], params.b[layer_id] = self.weights_initialization_strategy.initialize(layer)
        return params

    @staticmethod
    def _forward_propagation_step(A_prev, W_curr, b_curr, activation: activations.IActivationFunction):
        Z = np.dot(W_curr, A_prev) + b_curr
        return activation.calc(Z), Z

    def _forward_propagation(self, X):
        cache = models.CalculationsCache(A={}, Z={})

        A_curr = X
        for layer_id, layer in enumerate(self.layers, start=1):
            A_prev = A_curr

            activation = layer.activation
            W_curr = self.parameters.W[layer_id]
            b_curr = self.parameters.b[layer_id]
            A_curr, Z_curr = self._forward_propagation_step(
                A_prev=A_prev, W_curr=W_curr, b_curr=b_curr, activation=activation)

            cache.A[layer_id - 1] = A_prev
            cache.Z[layer_id] = Z_curr

        cache.A[len(self.layers)] = A_curr
        return A_curr, cache

    @staticmethod
    def _backward_propagation_step(dA_curr, W_curr, Z_curr, A_prev, activation: activations.IActivationFunction):
        m = A_prev.shape[1]

        dZ_curr = activation.derivative(dA_curr, Z_curr)
        dW_curr = np.dot(dZ_curr, A_prev.T) / m
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr

    def _backward_propagation(self, Y, cache: models.CalculationsCache):
        grads = models.Gradients(dW={}, db={})

        dA_prev = Y
        for layer_id, layer in reversed(list(enumerate(self.layers, start=1))):
            activation = layer.activation

            dA_curr = dA_prev

            A_prev = cache.A[layer_id - 1]
            Z_curr = cache.Z[layer_id]
            W_curr = self.parameters.W[layer_id]

            dA_prev, dW_curr, db_curr = self._backward_propagation_step(
                dA_curr=dA_curr, W_curr=W_curr, Z_curr=Z_curr, A_prev=A_prev, activation=activation)

            grads.dW[layer_id] = dW_curr
            grads.db[layer_id] = db_curr

        return grads

    def _epoch(self, trainDataSet: models.DataSet, epoch: int, eta: float, batches: int, optimizer: optimizers.IOptimizer):
        permutation = np.random.permutation(len(trainDataSet.labels))
        for batch in np.split(permutation, batches):
            X_batch = trainDataSet.X[batch].T
            Y_batch = trainDataSet.Y[batch].T
            
            optimizer.prepare(self.parameters)
            _, cache = self._forward_propagation(X_batch)

            grads = self._backward_propagation(Y=Y_batch, cache=cache)
            optimizer.update(epoch, eta, self.parameters, grads)

    @staticmethod
    def _cost(Y, Y_hat):
        return (- np.log(Y_hat) * Y).sum() / len(Y)

    def _val_epoch(self, trainDataSet: models.DataSet, devDataSet: models.DataSet):
        Y_train_hat = self.predict(X=trainDataSet.X)
        cost_train = self._cost(Y=trainDataSet.Y, Y_hat=Y_train_hat)
        accuracy_train = toolkit.accuracy(trainDataSet.labels, Y_train_hat)

        Y_dev_hat = self.predict(X=devDataSet.X)
        cost_dev = self._cost(Y=devDataSet.Y, Y_hat=Y_dev_hat)
        accuracy_val = toolkit.accuracy(devDataSet.labels, Y_dev_hat)

        return cost_train, cost_dev, accuracy_train, accuracy_val

    @staticmethod
    def _assert_model_is_not_overfitted(cost_dev: float, best_cost_dev: float, max_cost_rise: float):
        if cost_dev >= (1 + max_cost_rise) * best_cost_dev:
            raise EarlyStopping.overfitting()

    @staticmethod
    def _assert_model_accuracy_below_limit(accuracy_train: float, max_accuracy: float):
        if accuracy_train > max_accuracy:
            raise EarlyStopping.accuracy()

    @staticmethod
    def _assert_epoch_limit_is_not_exceed(epoch: int, epochs: int):
        if epoch == epochs:
            raise EarlyStopping.epochs()

    def fit2(self,
             trainDataSet: models.DataSet,
             devDataSet: models.DataSet,
             optimizer: optimizers.IOptimizer = optimizers.SGD(),
             epochs: int = 100,
             eta: float = 0.01,
             batch_size: int = 10,
             max_accuracy: float = 1,
             max_cost_rise: float = 0.1):

        W_copy, b_copy = models.copy(self.parameters.W, self.parameters.b)
        best = models.BestParameters(
            epoch=0,
            params=models.Parameters(W_copy, b_copy),
            cost_train=np.finfo('d').max,
            accuracy_train=np.finfo('d').max,
            cost_dev=np.finfo('d').max,
            accuracy_dev=np.finfo('d').max)
        stats = models.Statistics()

        batches = int(len(trainDataSet.labels) / batch_size)
        batches = batches if batches > 0 else 1

        epoch = 1
        while True:
            self._epoch(trainDataSet=trainDataSet, epoch=epoch, eta=eta, batches=batches, optimizer=optimizer)
            cost_train, cost_dev, accuracy_train, accuracy_val = self._val_epoch(
                trainDataSet=trainDataSet, devDataSet=devDataSet)

            print(
                f'Epoch={epoch}',
                f'CostTrain={cost_train:.3f}',
                f'AccuracyTrain={accuracy_train:.3f}',
                f'CostDev={cost_dev:.3f}',
                f'AccuracyDev={accuracy_val:.3f}')

            # Save stats
            stats.cost_history.cost_train.append(cost_train)
            stats.cost_history.cost_dev.append(cost_dev)
            stats.accuracy_history.accuracy_train.append(accuracy_train)
            stats.accuracy_history.accuracy_dev.append(accuracy_val)

            if cost_dev < best.cost_dev:
                W_copy, b_copy = models.copy(best.params.W, best.params.b)
                best = models.BestParameters(
                    epoch=epoch,
                    params=models.Parameters(W_copy, b_copy),
                    cost_train=cost_train,
                    accuracy_train=accuracy_train,
                    cost_dev=cost_dev,
                    accuracy_dev=accuracy_val)

            try:
                self._assert_model_is_not_overfitted(
                    best_cost_dev=best.cost_dev, cost_dev=cost_dev, max_cost_rise=max_cost_rise)
                self._assert_model_accuracy_below_limit(
                    accuracy_train=accuracy_train, max_accuracy=max_accuracy)
                self._assert_epoch_limit_is_not_exceed(
                    epoch=epoch, epochs=epochs)
            except EarlyStopping as e:
                W_copy, b_copy = models.copy(self.parameters.W, self.parameters.b)
                self.parameters = models.Parameters(W_copy, b_copy)
                return best, stats, e

            epoch += 1

    def predict(self, X):
        Y_hat, _ = self._forward_propagation(X.T)
        return Y_hat.T


def serialize_model(mlp: MultilayerPerceptron, filename):
    with open(filename, 'wb') as f:
        pickle.dump(mlp, f)


def deserialize_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
