import time

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
import numpy as np
from mnist import MNIST


def loadData(normalize=True, reshape=False):
    import pathlib
    data = MNIST(str(pathlib.Path(__file__).resolve().parent / 'data'))
    data.gz = True
    X, labels = data.load_training()
    X, labels = np.array(X, dtype="float64"), np.array(labels, dtype='float64')

    if normalize:
        X = X / 255

    if reshape:
        X = np.reshape(X, (X.shape[0], 28, 28, 1))

    return X, labels


def split(X, y, n=0.7, shuffle=False):
    assert n <= 1

    m = len(y)
    indices = np.random.permutation(len(y)) if shuffle else np.arange(len(y))

    indices1 = indices[:int(m*n)]
    indices2 = indices[int(m*n):]

    return X[indices1], y[indices1], X[indices2], y[indices2]


def buildModel(
        hiddenLayerNeuronsCount=128,
        hiddenLayerActivation='relu',
        kernelInitializer='he_uniform',
        convolutionLayer=None,
        poolingLayer=None):
    layers = []
    if convolutionLayer:
        layers.append(convolutionLayer)
    if poolingLayer:
        layers.append(poolingLayer)

    layers += [
        Flatten(),
        Dense(
            hiddenLayerNeuronsCount,
            activation=hiddenLayerActivation,
            kernel_initializer=kernelInitializer),
        Dense(10)
    ]

    return Sequential(layers)


def evaluate(model: Sequential, X_train, y_train, X_val, y_val, X_test, y_test, epochs=10, patience=5):
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-8,
        patience=patience,
        restore_best_weights=True)

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    results = {
        "epoch": [],
        "loss": [],
        "accuracy": [],
        "executionTime": []
    }

    N = 1
    for _ in range(N):
        history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), callbacks=[callback])

        start = time.time()
        loss, accuracy = model.evaluate(X_test, y_test, verbose=2)

        results["epoch"].append(len(history.epoch))
        results['loss'].append(loss)
        results['accuracy'].append(accuracy)
        results['executionTime'].append(time.time() - start)

    return {
        "epoch": sum(results['epoch']) / N,
        "loss": sum(results['loss']) / N,
        "accuracy": sum(results['accuracy']) / N,
        "executionTime": sum(results['executionTime']) / N
    }


def save(file, params, results):
    x = [f"{results['epoch']:.1f}", f"{results['accuracy']*100:.2f}\\%", f"{results['executionTime']:.2f}"]
    row = " && ".join(map(str, params)) + " && " + " && ".join(x) + " \\\\ \n"
    print(row)
    with open(file, 'a') as f:
        f.write(row)


def main():
    X, labels = loadData(normalize=True, reshape=True)
    X_train, y_train, X_test, y_test = split(X, labels, 0.7, shuffle=True)
    X_train, y_train, X_val, y_val = split(X_train, y_train, 0.9, shuffle=True)

    # Model Defaults
    hiddenLayerNeuronsCount = 128
    hiddenLayerActivation = 'relu'
    hiddenLayerKernelInitializer = 'he_uniform'

    # Evaluation defaults
    epochs = 20
    patience = 5

    print('REFERENCE')
    referenceModel = buildModel(
        hiddenLayerNeuronsCount=hiddenLayerNeuronsCount,
        hiddenLayerActivation=hiddenLayerActivation,
        kernelInitializer=hiddenLayerKernelInitializer,
        convolutionLayer=None,
        poolingLayer=None)
    results = evaluate(
        referenceModel, X_train, y_train, X_val, y_val, X_test, y_test, epochs=epochs, patience=patience)
    print(results)

    print('CONVOLUTION')
    bestFiltersParam, bestKernelSizeParam = 0, 0
    accuracy = 0.
    for kernelSize in [2, 3, 4, 5]:
        for filters in [4, 8, 12, 16, 20, 24, 28, 32]:
            print(f'CONVOLUTION: filters={filters} kernelSize={kernelSize}')
            subjectModel = buildModel(
                hiddenLayerNeuronsCount=hiddenLayerNeuronsCount,
                hiddenLayerActivation=hiddenLayerActivation,
                kernelInitializer=hiddenLayerKernelInitializer,
                convolutionLayer=Conv2D(filters=filters, kernel_size=kernelSize),
                poolingLayer=None)
            results = evaluate(
                subjectModel, X_train, y_train, X_val, y_val, X_test, y_test, epochs=epochs, patience=patience)
            if results["accuracy"] > accuracy:
                bestFiltersParam, bestKernelSizeParam = filters, kernelSize
            save("convolution.tex", [kernelSize, filters], results)

    print('CONVOLUTION + AVERAGE POOLING')
    for poolSize in [2, 3, 4, 5]:
        print(
            f'CONVOLUTION + AVERAGE POOLING: filters={bestFiltersParam} kernelSize={bestKernelSizeParam} '
            f'poolSize={poolSize}')
        subjectModel = buildModel(
            hiddenLayerNeuronsCount=hiddenLayerNeuronsCount,
            hiddenLayerActivation=hiddenLayerActivation,
            kernelInitializer=hiddenLayerKernelInitializer,
            convolutionLayer=Conv2D(filters=bestFiltersParam, kernel_size=bestKernelSizeParam),
            poolingLayer=AveragePooling2D(pool_size=poolSize))
        results = evaluate(
            subjectModel, X_train, y_train, X_val, y_val, X_test, y_test, epochs=epochs, patience=patience)
        save("convolution_avg_pooling.tex", [poolSize], results)

    print('CONVOLUTION + MAX POOLING')
    for poolSize in [2, 3, 4, 5]:
        print(
            f'CONVOLUTION + MAX POOLING: filters={bestFiltersParam} kernelSize={bestKernelSizeParam} '
            f'poolSize={poolSize}')
        subjectModel = buildModel(
            hiddenLayerNeuronsCount=hiddenLayerNeuronsCount,
            hiddenLayerActivation=hiddenLayerActivation,
            kernelInitializer=hiddenLayerKernelInitializer,
            convolutionLayer=Conv2D(filters=bestFiltersParam, kernel_size=bestKernelSizeParam),
            poolingLayer=MaxPooling2D(pool_size=poolSize))
        results = evaluate(
            subjectModel, X_train, y_train, X_val, y_val, X_test, y_test, epochs=epochs, patience=patience)
        save("convolution_max_pooling.tex", [poolSize], results)


if __name__ == '__main__':
    main()
