from abc import ABC, abstractmethod
from typing import List

import numpy as np

from . import models


class IOptimizer(ABC):

    @abstractmethod
    def prepare(self, parameters: models.Parameters):
        pass

    @abstractmethod
    def update(self, epoch: int, eta: float, parameters: models.Parameters, gradients: models.Gradients):
        pass


class SGD(IOptimizer):

    def prepare(self, parameters: models.Parameters):
        # Nothing is done here
        pass

    def update(self, epoch: int, eta: float, parameters: models.Parameters, gradients: models.Gradients):
        for layer_id in parameters.W:
            parameters.W[layer_id] -= eta * gradients.dW[layer_id]
            parameters.b[layer_id] -= eta * gradients.db[layer_id]


class Momentum(IOptimizer):

    def __init__(self, layers: List[models.Layer], mi=0.9):
        dW, db = models.from_layers(layers=layers, fillFunc=np.zeros)
        self.update_prev = models.Gradients(dW=dW, db=db)
        self.mi = mi

    def prepare(self, parameters: models.Parameters):
        # Nothing is done here
        pass

    def update(self, epoch: int, eta: float, parameters: models.Parameters, gradients: models.Gradients):
        update_curr = models.Gradients(dW={}, db={})
        for layer_id in parameters.W:
            update_curr.dW[layer_id] = eta * gradients.dW[layer_id] + self.mi * self.update_prev.dW[layer_id]
            update_curr.db[layer_id] = eta * gradients.db[layer_id] + self.mi * self.update_prev.db[layer_id]
            parameters.W[layer_id] -= update_curr.dW[layer_id]
            parameters.b[layer_id] -= update_curr.db[layer_id]
        self.update_prev = update_curr


class NesterovMomentum(IOptimizer):

    def __init__(self, layers: List[models.Layer], mi=0.9):
        dW, db = models.from_layers(layers=layers, fillFunc=np.zeros)
        self.update_prev = models.Gradients(dW=dW, db=db)
        self.mi = mi

    def prepare(self, parameters: models.Parameters):
        for layer_id in parameters.W:
            parameters.W[layer_id] -= self.mi * self.update_prev.dW[layer_id]
            parameters.b[layer_id] -= self.mi * self.update_prev.db[layer_id]

    def update(self, epoch: int, eta: float, parameters: models.Parameters, gradients: models.Gradients):
        update_curr = models.Gradients(dW={}, db={})
        for layer_id in parameters.W:
            update_curr.dW[layer_id] = eta * gradients.dW[layer_id] + self.mi * self.update_prev.dW[layer_id]
            update_curr.db[layer_id] = eta * gradients.db[layer_id] + self.mi * self.update_prev.db[layer_id]
            parameters.W[layer_id] -= update_curr.dW[layer_id]
            parameters.b[layer_id] -= update_curr.db[layer_id]
        self.update_prev = update_curr


class Adagrad(IOptimizer):

    def __init__(self, layers: List[models.Layer]):
        dW, db = models.from_layers(layers=layers, fillFunc=np.zeros)
        self.sumOfSquaredGradients = models.Gradients(dW=dW, db=db)

    def prepare(self, parameters: models.Parameters):
        # Nothing is done here
        pass

    def update(self, epoch: int, eta: float, parameters: models.Parameters, gradients: models.Gradients):
        for layer_id in parameters.W:
            self.sumOfSquaredGradients.dW[layer_id] += gradients.dW[layer_id] ** 2
            self.sumOfSquaredGradients.db[layer_id] += gradients.db[layer_id] ** 2

            parameters.W[layer_id] -= eta * gradients.dW[layer_id] / (
                    np.sqrt(self.sumOfSquaredGradients.dW[layer_id]) + 1e-8)
            parameters.b[layer_id] -= eta * gradients.db[layer_id] / (
                    np.sqrt(self.sumOfSquaredGradients.db[layer_id]) + 1e-8)


class AdaDelta(IOptimizer):

    def __init__(self, layers: List[models.Layer]):
        dW, db = models.from_layers(layers=layers, fillFunc=np.zeros)
        self.squaredGradients = models.Gradients(dW=dW, db=db)
        dW, db = models.from_layers(layers=layers, fillFunc=np.zeros)
        self.squaredGradients_prev = models.Gradients(dW=dW, db=db)
        dW, db = models.from_layers(layers=layers, fillFunc=np.zeros)
        self.gradients_prev = models.Gradients(dW=dW, db=db)

    def prepare(self, parameters: models.Parameters):
        # Nothing is done here
        pass

    def update(self, epoch: int, eta: float, parameters: models.Parameters, gradients: models.Gradients):
        for layer_id in parameters.W:
            # Weights

            self.squaredGradients.dW[layer_id] = 0.9 * self.squaredGradients.dW[layer_id] \
                                                 + 0.1 * gradients.dW[layer_id] ** 2

            rms_gW = np.sqrt(self.squaredGradients.dW[layer_id]) + 1e-8
            self.squaredGradients_prev.dW[layer_id] = 0.9 * self.squaredGradients_prev.dW[layer_id] \
                                                      + 0.1 * self.gradients_prev.dW[layer_id] ** 2

            rms_thetaW = np.sqrt(self.squaredGradients_prev.dW[layer_id]) + 1e-8
            parameters.W[layer_id] -= eta * np.divide(rms_thetaW, rms_gW) * gradients.dW[layer_id]

            # Biases
            self.squaredGradients.db[layer_id] = 0.9 * self.squaredGradients.db[layer_id] \
                                                 + 0.1 * gradients.db[layer_id] ** 2

            rms_gb = np.sqrt(self.squaredGradients.db[layer_id]) + 1e-8
            self.squaredGradients_prev.db[layer_id] = 0.9 * self.squaredGradients_prev.db[layer_id] \
                                                      + 0.1 * self.gradients_prev.db[layer_id] ** 2

            rms_thetab = np.sqrt(self.squaredGradients_prev.db[layer_id]) + 1e-8
            parameters.b[layer_id] -= eta * np.divide(rms_thetab, rms_gb) * gradients.db[layer_id]

            # Save current params
            self.gradients_prev.dW[layer_id] = gradients.dW[layer_id]
            self.gradients_prev.db[layer_id] = gradients.db[layer_id]


class Adam(IOptimizer):

    def __init__(self, layers: List[models.Layer], beta1=0.9, beta2=0.999, epsilon=1e-8):
        m_dW, m_db = models.from_layers(layers=layers, fillFunc=np.zeros)
        v_dW, v_db = models.from_layers(layers=layers, fillFunc=np.zeros)
        self.m = models.Gradients(dW=m_dW, db=m_db)
        self.v = models.Gradients(dW=v_dW, db=v_db)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def prepare(self, parameters: models.Parameters):
        # Nothing is done here
        pass

    def update(self, epoch: int, eta: float, parameters: models.Parameters, gradients: models.Gradients):
        for layer_id in parameters.W:
            self.m.dW[layer_id] = self.beta1 * self.m.dW[layer_id] + (1 - self.beta1) * gradients.dW[layer_id]
            self.m.db[layer_id] = self.beta1 * self.m.db[layer_id] + (1 - self.beta1) * gradients.db[layer_id]

            self.v.dW[layer_id] = self.beta2 * self.v.dW[layer_id] + (1 - self.beta2) * (gradients.dW[layer_id] ** 2)
            self.v.db[layer_id] = self.beta2 * self.v.db[layer_id] + (1 - self.beta2) * (gradients.db[layer_id] ** 2)

            # Bias correction
            m_dW_corr = self.m.dW[layer_id] / (1 - self.beta1 ** epoch)
            m_db_corr = self.m.db[layer_id] / (1 - self.beta1 ** epoch)
            v_dW_corr = self.v.dW[layer_id] / (1 - self.beta2 ** epoch)
            v_db_corr = self.v.db[layer_id] / (1 - self.beta2 ** epoch)

            parameters.W[layer_id] -= eta * (m_dW_corr / (np.sqrt(v_dW_corr) + self.epsilon))
            parameters.b[layer_id] -= eta * (m_db_corr / (np.sqrt(v_db_corr) + self.epsilon))
