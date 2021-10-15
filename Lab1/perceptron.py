import numpy as np


class Perceptron(object):

    def __init__(self, eta=0.01, threshold=0.0, gen=None):
        self.eta = eta
        self.threshold = threshold
        self.gen = gen
        self.epochs = None

    def _initialize_weights(self, m):
        self.w_ = np.asarray([np.float(self.gen()) if self.gen else np.float(0) for _ in range(m + 1)])
        self.w_initialized = True

    def _update_weights(self, x, label, bipolar):
        output = self.net_input(x)
        prediction = 1 if output > self.threshold else -1 if bipolar else 0
        error = label - prediction
        self.w_[0] += self.eta * error
        self.w_[1:] += self.eta * x.dot(error)
        return abs(error)

    def fit(self, X, y, bipolar=True, max_iter=10_000):
        self._initialize_weights(X.shape[1])

        iteration = 0
        errors = 1.0
        while errors > 0 and iteration < max_iter:
            cost = []
            for x, label in zip(X, y):
                cost.append(self._update_weights(x, label, bipolar))

            errors = sum(cost)
            iteration += 1
        self.epochs = iteration
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0] if self.threshold == 0.0 else 0

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X, bipolar=True):
        return np.where(self.activation(X) > self.threshold, 1, -1 if bipolar else 0)


def predict(X):
    X_train = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    y = np.array([-1, -1, -1, 1])

    perceptron = Perceptron(eta=0.01)
    perceptron.fit(X_train, y, bipolar=True)
    predictions = perceptron.predict(np.array(X), bipolar=True)

    print(predictions)
