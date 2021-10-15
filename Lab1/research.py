from random import uniform

from adaline import AdalineSGD
from perceptron import Perceptron


def score(predictions, y):
    acc = []
    for prediction, label in zip(predictions, y):
        acc.append(1 if prediction == label else 0)
    return sum(acc) / len(y)


def perceptron_threshold_bipolar(X, y, eta, w_range):
    thresholds = [threshold / 10 for threshold in range(-10, 10)]

    for threshold in thresholds:
        epochs = []
        for i in range(10):
            perceptron = Perceptron(eta=eta, threshold=threshold, gen=lambda: uniform(-w_range, w_range))
            perceptron.fit(X, y, bipolar=True)
            epochs.append(perceptron.epochs)
        print(threshold, sum(epochs) / 10, sep=' & ', end=' \\\\\n')


def perceptron_threshold_unipolar(X, y, eta, w_range):
    thresholds = [threshold / 10 for threshold in range(-10, 10)]

    for threshold in thresholds:
        epochs = []
        for i in range(10):
            perceptron = Perceptron(eta=eta, threshold=threshold, gen=lambda: uniform(-w_range, w_range))
            perceptron.fit(X, y, bipolar=False)
            epochs.append(perceptron.epochs)
        print(threshold, sum(epochs) / 10, sep=' & ', end=' \\\\\n')


def perceptron_w_range_bipolar(X, y, eta):
    w_ranges = [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.01,  0.001, 0.0001]

    for w_range in w_ranges:
        epochs = []
        for i in range(10):
            perceptron = Perceptron(eta=eta, threshold=0, gen=lambda: uniform(-w_range, w_range))
            perceptron.fit(X, y, bipolar=True)
            epochs.append(perceptron.epochs)
        print((-w_range, w_range), sum(epochs) / 10, sep=' & ', end=' \\\\\n')


def perceptron_w_range_unipolar(X, y, eta):
    w_ranges = [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.01,  0.001, 0.0001]

    for w_range in w_ranges:
        epochs = []
        for i in range(10):
            perceptron = Perceptron(eta=eta, threshold=0, gen=lambda: uniform(-w_range, w_range))
            perceptron.fit(X, y, bipolar=False)
            epochs.append(perceptron.epochs)
        print((-w_range, w_range), sum(epochs) / 10, sep=' & ', end=' \\\\\n')


def perceptron_eta_bipolar(X, y, w_range):
    etas = [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.01,  0.001, 0.0001]

    for eta in etas:
        epochs = []
        for i in range(10):
            perceptron = Perceptron(eta=eta, threshold=0, gen=lambda: uniform(-w_range, w_range))
            perceptron.fit(X, y, bipolar=True)
            epochs.append(perceptron.epochs)
        print(eta, sum(epochs) / 10, sep=' & ', end=' \\\\\n')


def perceptron_eta_unipolar(X, y, w_range):
    etas = [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.01,  0.001, 0.0001]

    for eta in etas:
        epochs = []
        for i in range(10):
            perceptron = Perceptron(eta=eta, threshold=0, gen=lambda: uniform(-w_range, w_range))
            perceptron.fit(X, y, bipolar=False)
            epochs.append(perceptron.epochs)
        print(eta, sum(epochs) / 10, sep=' & ', end=' \\\\\n')


def adaline_w_range_bipolar(X, y, eta, min_cost):
    w_ranges = [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.01, 0.001, 0.0001]

    for w_range in w_ranges:
        epochs = []
        scores = []
        for i in range(10):
            adaline = AdalineSGD(eta=eta, min_cost=min_cost, gen=lambda: uniform(-w_range, w_range))
            adaline.fit(X, y)
            epochs.append(adaline.epochs)
            scores.append(score(adaline.predict(X, bipolar=True), y))
        print((-w_range, w_range), sum(epochs) / 10, sum(scores) * 10, sep=' & ', end=' \\\\\n')


def adaline_w_range_unipolar(X, y, eta, min_cost):
    w_ranges = [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.01, 0.001, 0.0001]

    for w_range in w_ranges:
        epochs = []
        scores = []
        for i in range(10):
            adaline = AdalineSGD(eta=eta, min_cost=min_cost, gen=lambda: uniform(-w_range, w_range))
            adaline.fit(X, y)
            epochs.append(adaline.epochs)
            scores.append(score(adaline.predict(X, bipolar=False), y))
        print((-w_range, w_range), sum(epochs) / 10, sum(scores) * 10, sep=' & ', end=' \\\\\n')


def adaline_min_cost_bipolar(X, y, eta, w_range):
    min_costs = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    for min_cost in min_costs:
        epochs = []
        scores = []
        for i in range(10):
            adaline = AdalineSGD(eta=eta, min_cost=min_cost, gen=lambda: uniform(-w_range, w_range))
            adaline.fit(X, y)
            epochs.append(adaline.epochs)
            scores.append(score(adaline.predict(X, bipolar=True), y))
        print(min_cost, sum(epochs) / 10, sum(scores) * 10, sep=' & ', end=' \\\\\n')


def adaline_min_cost_unipolar(X, y, eta, w_range):
    min_costs = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.02, 0.01]

    for min_cost in min_costs:
        epochs = []
        scores = []
        for i in range(10):
            adaline = AdalineSGD(eta=eta, min_cost=min_cost, gen=lambda: uniform(-w_range, w_range))
            adaline.fit(X, y)
            epochs.append(adaline.epochs)
            scores.append(score(adaline.predict(X, bipolar=False), y))
        print(min_cost, sum(epochs) / 10, sum(scores) * 10, sep=' & ', end=' \\\\\n')


def adaline_eta_bipolar(X, y, min_cost, w_range):
    etas = [0.6, 0.4, 0.2, 0.1, 0.01,  0.001, 0.0001]

    for eta in etas:
        epochs = []
        scores = []
        for i in range(10):
            adaline = AdalineSGD(eta=eta, min_cost=min_cost, gen=lambda: uniform(-w_range, w_range))
            adaline.fit(X, y)
            epochs.append(adaline.epochs)
            scores.append(score(adaline.predict(X, bipolar=True), y))
        print(eta, sum(epochs) / 10, sum(scores) * 10, sep=' & ', end=' \\\\\n')


def adaline_eta_unipolar(X, y, min_cost, w_range):
    etas = [0.6, 0.4, 0.2, 0.1, 0.01,  0.001, 0.0001]

    for eta in etas:
        epochs = []
        scores = []
        for i in range(10):
            adaline = AdalineSGD(eta=eta, min_cost=min_cost, gen=lambda: uniform(-w_range, w_range))
            adaline.fit(X, y)
            epochs.append(adaline.epochs)
            scores.append(score(adaline.predict(X, bipolar=False), y))
        print(eta, sum(epochs) / 10, sum(scores) * 10, sep=' & ', end=' \\\\\n')