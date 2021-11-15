import asyncio
import sys
import pathlib
from datetime import datetime

path = str(pathlib.Path(__file__).resolve().parent.parent)
sys.path.append(path)

import numpy as np
from dataclasses import dataclass
from mnist import MNIST
from Lab2.mlp2 import Layer, MultilayerPerceptron
from Lab2.mltoolkit import Softmax, Relu, Tanh, IActivationFunction, DataSet, accuracy, Stats
from Lab2.workers import WorkerPool, create_job


@dataclass
class Defaults:
    mean: float
    sigma: float
    outputs: int
    epochs: int
    eta: float
    batch_size: int
    max_train_accuracy: float
    max_cost_dev_rise: float
    N_train: float
    N_dev: float
    N_test: float


class ParalleledResearch:

    def __init__(self, train_set: DataSet, dev_set: DataSet, test_set: DataSet, defaults: Defaults, parallel=2):
        self.train_set = train_set
        self.dev_set = dev_set
        self.test_set = test_set
        self.defaults = defaults
        self.parallel = parallel

    async def research1_relu(self):
        await self._research1(title='Research 1 - ReLu', activation=Relu)

    async def research1_tanh(self):
        await self._research1(title='Research 1 - TanH', activation=Tanh)

    async def research2_relu(self):
        await self._research2(title='Research 2 - ReLu', activation=Relu)

    async def research2_tanh(self):
        await self._research2(title='Research 2 - TanH', activation=Tanh)

    async def research3_relu(self):
        await self._research3(title='Research 3 - ReLu', activation=Relu)

    async def research3_tanh(self):
        await self._research3(title='Research 3 - TanH', activation=Tanh)

    async def research4_relu(self):
        await self._research4(title='Research 4 - ReLu', activation=Relu)

    async def research4_tanh(self):
        await self._research4(title='Research 4 - TanH', activation=Tanh)

    async def _research1(self, title, activation):
        accuracies = []
        epochs = []
        for outputs in [1, 5, 20, 50, 100, 200, 500, 1000, 2000]:
            accuracy_, epoch = await self._do_research(
                await self._create_research1_jobs(outputs=outputs, activation=activation))
            accuracies.append(accuracy_)
            epochs.append(epoch)
            print(f"Hidden layer size={outputs}", f"Activation={activation}",
                  f'Avg. accuracy={accuracy_}', f'Avg. epoch={epoch}')
        print(title)
        print(f'Accuracy={accuracies}')
        print(f'Epoch={epochs}')

    async def _create_research1_jobs(self, outputs, activation):
        jobs = []
        for i in range(10):
            layers = create_one_hidden_layer(self.train_set.X.shape[1], outputs=outputs, activation=activation())

            job = create_job(
                target=self._fit_and_verify, layers=layers, mean=self.defaults.mean, sigma=self.defaults.sigma,
                epochs=self.defaults.epochs, eta=self.defaults.eta, batch_size=self.defaults.batch_size,
                max_train_accuracy=self.defaults.max_train_accuracy, max_cost_dev_rise=self.defaults.max_cost_dev_rise)

            jobs.append(job)
        return jobs

    async def _research2(self, title, activation):
        accuracies = []
        epochs = []
        for eta in [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]:
            accuracy_, epoch = await self._do_research(
                await self._create_research2_jobs(eta=eta, activation=activation))
            accuracies.append(accuracy_)
            epochs.append(epoch)
            print(f"Eta={eta}", f"Activation={activation}",
                  f'Avg. accuracy={accuracy_}', f'Avg. epoch={epoch}')
        print(title)
        print(f'Accuracy={accuracies}')
        print(f'Epoch={epochs}')

    async def _create_research2_jobs(self, eta, activation):
        jobs = []
        for i in range(10):
            layers = create_one_hidden_layer(
                self.train_set.X.shape[1], outputs=self.defaults.outputs, activation=activation())

            job = create_job(
                target=self._fit_and_verify, layers=layers, mean=self.defaults.mean, sigma=self.defaults.sigma,
                epochs=self.defaults.epochs, eta=eta, batch_size=self.defaults.batch_size,
                max_train_accuracy=self.defaults.max_train_accuracy, max_cost_dev_rise=self.defaults.max_cost_dev_rise)

            jobs.append(job)
        return jobs

    async def _research3(self, title, activation):
        N = self.train_set.X.shape[0]

        accuracies = []
        epochs = []
        for batch_size in [N, N / 2, 1000, 100, 10, 1]:
            accuracy_, epoch = await self._do_research(
                await self._create_research3_jobs(batch_size=batch_size, activation=activation))
            accuracies.append(accuracy_)
            epochs.append(epoch)
            print(f"Batch size={batch_size}", f"Activation={activation}",
                  f'Avg. accuracy={accuracy_}', f'Avg. epoch={epoch}')

        print(title)
        print(f'Accuracy={accuracies}')
        print(f'Epoch={epochs}')

    async def _create_research3_jobs(self, batch_size, activation):
        jobs = []
        for i in range(10):
            layers = create_one_hidden_layer(
                self.train_set.X.shape[1], outputs=self.defaults.outputs, activation=activation())

            job = create_job(
                target=self._fit_and_verify, layers=layers, mean=self.defaults.mean, sigma=self.defaults.sigma,
                epochs=self.defaults.epochs, eta=self.defaults.eta, batch_size=batch_size,
                max_train_accuracy=self.defaults.max_train_accuracy, max_cost_dev_rise=self.defaults.max_cost_dev_rise)

            jobs.append(job)
        return jobs

    async def _research4(self, title, activation):
        accuracies = []
        epochs = []
        for sigma in [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]:
            accuracy_, epoch = await self._do_research(
                await self._create_research4_jobs(sigma=sigma, activation=activation))
            accuracies.append(accuracy_)
            epochs.append(epoch)
            print(f"Sigma={sigma}", f"Activation={activation}",
                  f'Avg. accuracy={accuracy_}', f'Avg. epoch={epoch}')
        print(title)
        print(f'Accuracy={accuracies}')
        print(f'Epoch={epochs}')

    async def _create_research4_jobs(self, sigma, activation):
        jobs = []
        for i in range(10):
            layers = create_one_hidden_layer(
                self.train_set.X.shape[1], outputs=self.defaults.outputs, activation=activation())

            job = create_job(
                target=self._fit_and_verify, layers=layers, mean=self.defaults.mean, sigma=sigma,
                epochs=self.defaults.epochs, eta=self.defaults.eta, batch_size=self.defaults.batch_size,
                max_train_accuracy=self.defaults.max_train_accuracy, max_cost_dev_rise=self.defaults.max_cost_dev_rise)

            jobs.append(job)
        return jobs

    def _fit_and_verify(self, layers, mean, sigma, batch_size, epochs, eta, max_train_accuracy, max_cost_dev_rise):
        mlp = MultilayerPerceptron(layers=layers, mean=mean, sigma=sigma)
        best, stats, reason = mlp.fit2(
            train_data_set=self.train_set, dev_data_set=self.dev_set, epochs=epochs, eta=eta, batch_size=batch_size,
            max_train_accuracy=max_train_accuracy, max_cost_dev_rise=max_cost_dev_rise)

        Y_test_hat = mlp.predict(X=self.test_set.X)
        acc = accuracy(labels=self.test_set.labels, Y_hat=Y_test_hat)

        return acc, best.epoch, reason

    async def _do_research(self, jobs):
        pool = WorkerPool(self.parallel)
        results = await pool.dispatch(jobs)
        pool.dispose()
        return self._unpack_fit_job_results(results)

    @staticmethod
    def _unpack_fit_job_results(results):
        accuracies = []
        epochs = []
        for acc, epoch, _ in results:
            accuracies.append(acc)
            epochs.append(epoch)

        return np.average(accuracies), np.average(epochs)


def load_data_set():
    import pathlib
    data = MNIST(str(pathlib.Path(__file__).resolve().parent / 'data'))
    data.gz = True
    X, Y = data.load_training()
    X = np.array(X)
    Y = np.array(Y)
    return np.array(X), np.array(Y)


def prepare_data_set():
    X, labels = load_data_set()
    X = X / 255
    Y = np.zeros(shape=(labels.shape[0], 10))
    for i in range(labels.shape[0]):
        Y[i][labels[i]] = 1

    return DataSet(X=X, Y=Y, labels=labels)


def split(data_set: DataSet, train, dev, test):
    assert train + dev + test <= 1

    N = len(data_set.labels)
    N_train = int(train * N)
    N_dev = int(N_train + dev * N)
    N_test = int(N_dev + test * N)

    train_data_set = DataSet(
        X=data_set.X[:N_train], Y=data_set.Y[:N_train], labels=data_set.labels[:N_train])
    dev_data_set = DataSet(
        X=data_set.X[N_train:N_dev], Y=data_set.Y[N_train:N_dev], labels=data_set.labels[N_train:N_dev])
    test_data_set = DataSet(
        X=data_set.X[N_dev:N_test], Y=data_set.Y[N_dev:N_test], labels=data_set.labels[N_dev:N_test])

    return train_data_set, dev_data_set, test_data_set


def create_one_hidden_layer(inputs: int, outputs: int, activation: IActivationFunction):
    return [
        Layer(inputs=inputs, outputs=outputs, activation=activation),
        Layer(inputs=outputs, outputs=10, activation=Softmax())
    ]


def plot(stats: Stats):
    epochs = [epoch for epoch in range(len(stats.cost_train_history))]
    import matplotlib.pyplot as plt
    plt.plot(epochs, stats.cost_train_history, 'r')
    plt.plot(epochs, stats.cost_dev_history, 'b')
    plt.show()


async def main():
    data_set = prepare_data_set()

    defaults = Defaults(
        mean=0.0,
        sigma=0.01,
        outputs=100,
        epochs=100,
        eta=0.01,
        batch_size=100,
        max_train_accuracy=0.95,
        max_cost_dev_rise=0.1,
        N_train=0.5,
        N_dev=0.1,
        N_test=0.1
    )

    train_data_set, dev_data_set, test_data_set = split(
        data_set=data_set, train=defaults.N_train, dev=defaults.N_dev, test=defaults.N_test)

    research = ParalleledResearch(
        train_set=train_data_set, dev_set=dev_data_set, test_set=test_data_set,
        defaults=defaults)

    print('Research 1', datetime.now())
    task1 = asyncio.create_task(research.research1_relu())
    task2 = asyncio.create_task(research.research1_tanh())
    await task1
    await task2

    print('Research 2', datetime.now())
    task1 = asyncio.create_task(research.research2_relu())
    task2 = asyncio.create_task(research.research2_tanh())
    await task1
    await task2

    print('Research 3', datetime.now())
    task1 = asyncio.create_task(research.research3_relu())
    task2 = asyncio.create_task(research.research3_tanh())
    await task1
    await task2

    print('Research 4', datetime.now())
    task1 = asyncio.create_task(research.research4_relu())
    task2 = asyncio.create_task(research.research4_tanh())
    await task1
    await task2

    print('Done :)', datetime.now())

if __name__ == '__main__':
    np.random.seed(909)
    asyncio.get_event_loop().run_until_complete(main())
