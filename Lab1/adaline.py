import numpy as np


class AdalineSGD(object):

    def __init__(self, eta=0.01, min_cost=0.5, gen=None):
        self.eta = eta
        self.min_cost = min_cost
        self.epochs = None
        self.gen = gen

    def _initialize_weights(self, m):
        self.w_ = np.asarray([np.float(self.gen()) if self.gen else np.float(0) for _ in range(m + 1)])

    def _update_weights(self, x, label):
        output = self.net_input(x)
        error = label - output
        self.w_[0] += self.eta * error
        self.w_[1:] += self.eta * x.dot(error)
        squared_error = error ** 2
        return squared_error

    def fit(self, X, y, max_iter=10_000):
        self._initialize_weights(X.shape[1])

        iteration = 0
        avg_cost = 10e20
        while avg_cost > self.min_cost and iteration < max_iter:
            cost = []
            for x, label in zip(X, y):
                cost.append(self._update_weights(x, label))

            avg_cost = sum(cost) / len(y)
            iteration += 1
        self.epochs = iteration
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X, bipolar=True):
        return np.where(self.activation(X) > 0.0, 1, -1 if bipolar else 0)


def predict(X):
    X_train = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    y = np.array([-1, -1, -1, 1])

    adaline = AdalineSGD(min_cost=0.3)
    adaline.fit(X_train, y, max_iter=1_000_000)
    predictions = adaline.predict(X, bipolar=True)

    print(predictions)
