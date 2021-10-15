import sys
import numpy as np
import adaline
import perceptron
from research import *


def main():
    X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    y = np.array([-1, -1, -1, 1])

    X_ = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_ = np.array([0, 0, 0, 1])

    # Comment out line you do not want to run
    # perceptron_threshold_bipolar(X, y, eta=0.01, w_range=0.01)
    # perceptron_threshold_unipolar(X_, y_, eta=0.01, w_range=0.01)
    # perceptron_w_range_bipolar(X, y, eta=0.01)
    # perceptron_w_range_unipolar(X_, y_, eta=0.01)
    # perceptron_eta_bipolar(X, y, w_range=0.3)
    # perceptron_eta_unipolar(X_, y_, w_range=0.3)
    # adaline_w_range_bipolar(X, y, eta=0.01, min_cost=0.5)
    # adaline_w_range_unipolar(X_, y_, eta=0.01, min_cost=0.1)
    # adaline_eta_bipolar(X, y, min_cost=0.5, w_range=0.5)
    # adaline_eta_unipolar(X_, y_, min_cost=0.1, w_range=0.5)
    # adaline_min_cost_bipolar(X, y, eta=0.01, w_range=0.5)
    # adaline_min_cost_unipolar(X_, y_, eta=0.01, w_range=0.01)

    pass


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
        exit(0)

    try:
        if sys.argv[1] == 'perceptron':
            perceptron.predict(np.array([float(sys.argv[2]), float(sys.argv[3])]))
        elif sys.argv[1] == 'adaline':
            adaline.predict(np.array([float(sys.argv[2]), float(sys.argv[3])]))
    except Exception:
        print("See README.MD for help.")
