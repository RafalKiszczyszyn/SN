import numpy as np
from mnist import MNIST

from Lab2 import models


def loadMnistDataSet() -> models.DataSet:
    import pathlib
    data = MNIST(str(pathlib.Path(__file__).resolve().parent / 'data'))
    data.gz = True
    X, labels = data.load_training()
    X, labels = np.array(X), np.array(labels)

    # normalize
    X = X / 255
    Y = np.zeros(shape=(labels.shape[0], 10))
    for i in range(labels.shape[0]):
        Y[i][labels[i]] = 1

    return models.DataSet(X=X, Y=Y, labels=labels)


def split(data_set: models.DataSet, train: float, dev: float, test: float):
    assert train + dev + test <= 1

    N = len(data_set.labels)
    N_train = int(train * N)
    N_dev = int(N_train + dev * N)
    N_test = int(N_dev + test * N)

    train_data_set = models.DataSet(
        X=data_set.X[:N_train], Y=data_set.Y[:N_train], labels=data_set.labels[:N_train])
    dev_data_set = models.DataSet(
        X=data_set.X[N_train:N_dev], Y=data_set.Y[N_train:N_dev], labels=data_set.labels[N_train:N_dev])
    test_data_set = models.DataSet(
        X=data_set.X[N_dev:N_test], Y=data_set.Y[N_dev:N_test], labels=data_set.labels[N_dev:N_test])

    return train_data_set, dev_data_set, test_data_set


def accuracy(labels: np.ndarray, Y_hat: np.ndarray):
    predicted_labels = Y_hat.argmax(axis=1)
    matches = predicted_labels == labels
    correct = matches.sum()
    return correct / len(labels)
