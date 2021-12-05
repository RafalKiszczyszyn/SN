from __future__ import annotations

import sys
import pathlib
import numpy as np

path = str(pathlib.Path(__file__).resolve().parent.parent)
sys.path.append(path)
from Lab2 import models, activations, optimizers, toolkit
from Lab2.mlp import MultilayerPerceptron
from Lab2.weights_initialization import RandomWeightsInitializationStrategy, \
    XavierWeightsInitializationStrategy, HeWeightsInitializationStrategy


# LOGGING LEVELS
HIGH = 1
LOW = 2

N = 10
EPOCHS = 50
ETA = 0.01
BATCH_SIZE = 100
MAX_TRAIN_ACCURACY = 0.95
MAX_TRAIN_ACCURACY2 = 0.99
MAX_COST_RISE = 0.1


def log(level, message):
    print(message)
    message += '\n'
    if level in [HIGH, LOW]:
        with open('DetailedResults.txt', 'a') as f:
            f.write(message)
    if level in [HIGH]:
        with open('Results.txt', 'a') as f:
            f.write(message)


def run(_id, model_factory_method, trainDataSet, devDataSet, testDataSet, *args, **kwargs):
    accuracy = []
    epochs = []

    for i in range(N):
        model = model_factory_method()
        best, stats, reason = model.fit2(trainDataSet=trainDataSet, devDataSet=devDataSet, *args, **kwargs)

        Y_hat = model.predict(X=testDataSet.X)
        accuracy_test = toolkit.accuracy(testDataSet.labels, Y_hat)

        epochs.append(best.epoch)
        accuracy.append(accuracy_test)

        log(LOW, f'Id="{_id}" Run={i + 1} Epoch={best.epoch} Accuracy={accuracy_test * 100:.2f}')

    log(HIGH, f'Id="{_id}" Epoch={sum(epochs) / N:.2f} Accuracy={sum(accuracy) / N * 100:.2f}')


def optimizers_test(_id, layers, model_factory_method, trainDataSet, devDataSet, testDataSet):

    run(_id=f'{_id}: Optimizer=None, Eta=0.001',
        model_factory_method=model_factory_method,
        trainDataSet=trainDataSet,
        devDataSet=devDataSet,
        testDataSet=testDataSet,
        optimizer=optimizers.SGD(),
        epochs=EPOCHS,
        eta=0.001,
        batch_size=BATCH_SIZE,
        max_accuracy=MAX_TRAIN_ACCURACY,
        max_cost_rise=MAX_COST_RISE)

    run(_id=f'{_id}: Optimizer=None, Eta=0.01',
        model_factory_method=model_factory_method,
        trainDataSet=trainDataSet,
        devDataSet=devDataSet,
        testDataSet=testDataSet,
        optimizer=optimizers.SGD(),
        epochs=EPOCHS,
        eta=0.01,
        batch_size=BATCH_SIZE,
        max_accuracy=MAX_TRAIN_ACCURACY,
        max_cost_rise=MAX_COST_RISE)

    run(_id=f'{_id}: Optimizer=None, Eta=0.1',
        model_factory_method=model_factory_method,
        trainDataSet=trainDataSet,
        devDataSet=devDataSet,
        testDataSet=testDataSet,
        optimizer=optimizers.SGD(),
        epochs=EPOCHS,
        eta=0.1,
        batch_size=BATCH_SIZE,
        max_accuracy=MAX_TRAIN_ACCURACY,
        max_cost_rise=MAX_COST_RISE)

    run(_id=f'{_id}: Optimizer=Momentum, Eta=0.001',
        model_factory_method=model_factory_method,
        trainDataSet=trainDataSet,
        devDataSet=devDataSet,
        testDataSet=testDataSet,
        optimizer=optimizers.Momentum(layers),
        epochs=EPOCHS,
        eta=0.001,
        batch_size=BATCH_SIZE,
        max_accuracy=MAX_TRAIN_ACCURACY,
        max_cost_rise=MAX_COST_RISE)

    run(_id=f'{_id}: Optimizer=Momentum, Eta=0.01',
        model_factory_method=model_factory_method,
        trainDataSet=trainDataSet,
        devDataSet=devDataSet,
        testDataSet=testDataSet,
        optimizer=optimizers.Momentum(layers),
        epochs=EPOCHS,
        eta=0.01,
        batch_size=BATCH_SIZE,
        max_accuracy=MAX_TRAIN_ACCURACY,
        max_cost_rise=MAX_COST_RISE)

    run(_id=f'{_id}: Optimizer=Momentum, Eta=0.1',
        model_factory_method=model_factory_method,
        trainDataSet=trainDataSet,
        devDataSet=devDataSet,
        testDataSet=testDataSet,
        optimizer=optimizers.Momentum(layers),
        epochs=EPOCHS,
        eta=0.1,
        batch_size=BATCH_SIZE,
        max_accuracy=MAX_TRAIN_ACCURACY,
        max_cost_rise=MAX_COST_RISE)

    run(_id=f'{_id}: Optimizer=NesterovMomentum, Eta=0.001',
        model_factory_method=model_factory_method,
        trainDataSet=trainDataSet,
        devDataSet=devDataSet,
        testDataSet=testDataSet,
        optimizer=optimizers.NesterovMomentum(layers),
        epochs=EPOCHS,
        eta=0.001,
        batch_size=BATCH_SIZE,
        max_accuracy=MAX_TRAIN_ACCURACY,
        max_cost_rise=MAX_COST_RISE)

    run(_id=f'{_id}: Optimizer=NesterovMomentum, Eta=0.01',
        model_factory_method=model_factory_method,
        trainDataSet=trainDataSet,
        devDataSet=devDataSet,
        testDataSet=testDataSet,
        optimizer=optimizers.NesterovMomentum(layers),
        epochs=EPOCHS,
        eta=0.01,
        batch_size=BATCH_SIZE,
        max_accuracy=MAX_TRAIN_ACCURACY,
        max_cost_rise=MAX_COST_RISE)

    run(_id=f'{_id}: Optimizer=NesterovMomentum, Eta=0.1',
        model_factory_method=model_factory_method,
        trainDataSet=trainDataSet,
        devDataSet=devDataSet,
        testDataSet=testDataSet,
        optimizer=optimizers.NesterovMomentum(layers),
        epochs=EPOCHS,
        eta=0.1,
        batch_size=BATCH_SIZE,
        max_accuracy=MAX_TRAIN_ACCURACY,
        max_cost_rise=MAX_COST_RISE)

    run(_id=f'{_id}: Optimizer=AdaGrad',
        model_factory_method=model_factory_method,
        trainDataSet=trainDataSet,
        devDataSet=devDataSet,
        testDataSet=testDataSet,
        optimizer=optimizers.Adagrad(layers),
        epochs=EPOCHS,
        eta=ETA,
        batch_size=BATCH_SIZE,
        max_accuracy=MAX_TRAIN_ACCURACY,
        max_cost_rise=MAX_COST_RISE)

    run(_id=f'{_id}: Optimizer=AdaDelta',
        model_factory_method=model_factory_method,
        trainDataSet=trainDataSet,
        devDataSet=devDataSet,
        testDataSet=testDataSet,
        optimizer=optimizers.AdaDelta(layers),
        epochs=EPOCHS,
        eta=ETA,
        batch_size=BATCH_SIZE,
        max_accuracy=MAX_TRAIN_ACCURACY,
        max_cost_rise=MAX_COST_RISE)

    run(_id=f'{_id}: Optimizer=Adam',
        model_factory_method=model_factory_method,
        trainDataSet=trainDataSet,
        devDataSet=devDataSet,
        testDataSet=testDataSet,
        optimizer=optimizers.Adam(layers),
        epochs=EPOCHS,
        eta=ETA,
        batch_size=BATCH_SIZE,
        max_accuracy=MAX_TRAIN_ACCURACY,
        max_cost_rise=MAX_COST_RISE)


def initializations_test(_id, layers, trainDataSet, devDataSet, testDataSet):

    run(_id=f'{_id}: Initialization Strategy=None, Simga=0.001',
        model_factory_method=lambda: MultilayerPerceptron(
            layers=layers, weights_initialization_strategy=RandomWeightsInitializationStrategy(sigma=0.001)),
        trainDataSet=trainDataSet,
        devDataSet=devDataSet,
        testDataSet=testDataSet,
        optimizer=optimizers.Adam(layers),
        epochs=EPOCHS,
        eta=ETA,
        batch_size=BATCH_SIZE,
        max_accuracy=MAX_TRAIN_ACCURACY2,
        max_cost_rise=MAX_COST_RISE)

    run(_id=f'{_id}: Initialization Strategy=None, Simga=0.01',
        model_factory_method=lambda: MultilayerPerceptron(
            layers=layers, weights_initialization_strategy=RandomWeightsInitializationStrategy(sigma=0.01)),
        trainDataSet=trainDataSet,
        devDataSet=devDataSet,
        testDataSet=testDataSet,
        optimizer=optimizers.Adam(layers),
        epochs=EPOCHS,
        eta=ETA,
        batch_size=BATCH_SIZE,
        max_accuracy=MAX_TRAIN_ACCURACY2,
        max_cost_rise=MAX_COST_RISE)

    run(_id=f'{_id}: Initialization Strategy=None, Simga=0.1',
        model_factory_method=lambda: MultilayerPerceptron(
            layers=layers, weights_initialization_strategy=RandomWeightsInitializationStrategy(sigma=0.1)),
        trainDataSet=trainDataSet,
        devDataSet=devDataSet,
        testDataSet=testDataSet,
        optimizer=optimizers.Adam(layers),
        epochs=EPOCHS,
        eta=ETA,
        batch_size=BATCH_SIZE,
        max_accuracy=MAX_TRAIN_ACCURACY2,
        max_cost_rise=MAX_COST_RISE)

    run(_id=f'{_id}: Initialization Strategy=None, Simga=0.5',
        model_factory_method=lambda: MultilayerPerceptron(
            layers=layers, weights_initialization_strategy=RandomWeightsInitializationStrategy(sigma=0.5)),
        trainDataSet=trainDataSet,
        devDataSet=devDataSet,
        testDataSet=testDataSet,
        optimizer=optimizers.Adam(layers),
        epochs=EPOCHS,
        eta=ETA,
        batch_size=BATCH_SIZE,
        max_accuracy=MAX_TRAIN_ACCURACY2,
        max_cost_rise=MAX_COST_RISE)

    run(_id=f'{_id}: Initialization Strategy=None, Simga=0.9',
        model_factory_method=lambda: MultilayerPerceptron(
            layers=layers, weights_initialization_strategy=RandomWeightsInitializationStrategy(sigma=0.9)),
        trainDataSet=trainDataSet,
        devDataSet=devDataSet,
        testDataSet=testDataSet,
        optimizer=optimizers.Adam(layers),
        epochs=EPOCHS,
        eta=ETA,
        batch_size=BATCH_SIZE,
        max_accuracy=MAX_TRAIN_ACCURACY2,
        max_cost_rise=MAX_COST_RISE)

    run(_id=f'{_id}: Initialization Strategy=Xaxier',
        model_factory_method=lambda: MultilayerPerceptron(
            layers=layers, weights_initialization_strategy=XavierWeightsInitializationStrategy()),
        trainDataSet=trainDataSet,
        devDataSet=devDataSet,
        testDataSet=testDataSet,
        optimizer=optimizers.Adam(layers),
        epochs=EPOCHS,
        eta=ETA,
        batch_size=BATCH_SIZE,
        max_accuracy=MAX_TRAIN_ACCURACY2,
        max_cost_rise=MAX_COST_RISE)

    run(_id=f'{_id}: Initialization Strategy=He',
        model_factory_method=lambda: MultilayerPerceptron(
            layers=layers, weights_initialization_strategy=HeWeightsInitializationStrategy()),
        trainDataSet=trainDataSet,
        devDataSet=devDataSet,
        testDataSet=testDataSet,
        optimizer=optimizers.Adam(layers),
        epochs=EPOCHS,
        eta=ETA,
        batch_size=BATCH_SIZE,
        max_accuracy=MAX_TRAIN_ACCURACY2,
        max_cost_rise=MAX_COST_RISE)


def main():
    np.random.seed(909)

    dataSet = toolkit.loadMnistDataSet()
    trainDataSet, devDataSet, testDataSet = toolkit.split(dataSet, 0.8, 0.1, 0.1)

    sigmoidArchitecture = [
        models.Layer(inputs=trainDataSet.X.shape[1], outputs=80, activation=activations.Sigmoid()),
        models.Layer(inputs=80, outputs=10, activation=activations.Softmax())
    ]

    reluArchitecture = [
        models.Layer(inputs=trainDataSet.X.shape[1], outputs=80, activation=activations.Relu()),
        models.Layer(inputs=80, outputs=10, activation=activations.Softmax())
    ]

    optimizers_test(
        'SIGMOID',
        sigmoidArchitecture,
        lambda: MultilayerPerceptron(
            layers=sigmoidArchitecture, weights_initialization_strategy=RandomWeightsInitializationStrategy()),
        trainDataSet, devDataSet, testDataSet)

    optimizers_test(
        'RELU',
        reluArchitecture,
        lambda: MultilayerPerceptron(
            layers=reluArchitecture, weights_initialization_strategy=RandomWeightsInitializationStrategy()),
        trainDataSet, devDataSet, testDataSet)

    initializations_test("SIGMOID", sigmoidArchitecture, trainDataSet, devDataSet, testDataSet)
    initializations_test("RELU", reluArchitecture, trainDataSet, devDataSet, testDataSet)


if __name__ == '__main__':
    main()
