import numpy as np

def sigmoid(z):
    return 1./(1. + np.e**(-z))

def sigmoid_prim(z):
    sigmoidz = sigmoid(z)
    return sigmoidz * (1 - sigmoidz)

def tanh(z):
    return 2./(1. + np.e**(-2.*z)) - 1.

def tanh_prim(z):
    tanhz = tanh(z)
    return 1. - tanhz**2

def relu(z):
    return np.minimum(np.maximum(0, z), 5)

def relu_prim(z):
    return (z > 0).astype(int)

def softmax(z):
    expz = np.exp(z - np.max(z))
    return expz / np.sum(expz, axis=0)

def softmax_prim(z):
    softmaxz = softmax(z)
    return softmaxz * (1 - softmaxz)

class MLP:

    def __init__(self, neuron_count_array, initialize_sigma, activation_function, alpha_optimizer = None, conv: bool = False, filter_size = 3):
        self.conv = conv
        self.filter_size = filter_size
        self.filter_count = 32
        self.filterW = np.random.randn(self.filter_count, filter_size, filter_size) / filter_size ** 2
        self.sizes = neuron_count_array
        self.W, self.b = self.initialize_weights(neuron_count_array, initialize_sigma)
        if activation_function == 'Sigmoid':
            self.activation_function = sigmoid
            self.activation_function_prim = sigmoid_prim
        elif activation_function == 'Tanh':
            self.activation_function = tanh
            self.activation_function_prim = tanh_prim
        elif activation_function == 'Relu':
            self.activation_function = relu
            self.activation_function_prim = relu_prim
        else:
            raise Exception('Wrong activation function!')

        self.best_W = self.W
        self.best_b = self.b
        self.best_val_accuracy = 0
        self.alpha_optimizer = alpha_optimizer
        #Momentum
        self.previous_delta_W = None
        self.previous_delta_b = None
        #Adagrad
        self.squared_deltas_W = None
        self.squared_deltas_b = None
        #Adadelta
        self.squared_previous_W = None
        self.squared_previous_b = None
        #Adam
        self.mW = None
        self.mb = None
        self.vW = None
        self.vb = None
        #values for conv layers
        self.previous_conv_X: np.ndarray
        self.previous_max_pooling_X: np.ndarray
        self.before_flatten_shape = (0, 0, 0)
        self.deltaA3: np.ndarray

    def initialize_weights(self, neuron_count_array, sigma):
        if sigma == 'Xavier':
            W = [np.sqrt(2 / (neuron_count_array[i + 1] + neuron_count_array[i])) * np.random.randn(neuron_count_array[i + 1], neuron_count_array[i]) for i in range(len(neuron_count_array) - 1)]
            b = [0 * np.random.randn(neuron_count_array[i + 1], 1) for i in range(len(neuron_count_array) - 1)]
        elif sigma == 'He':
            W = [np.sqrt(2 / (neuron_count_array[i])) * np.random.randn(neuron_count_array[i + 1], neuron_count_array[i]) for i in range(len(neuron_count_array) - 1)]
            b = [0 * np.random.randn(neuron_count_array[i + 1], 1) for i in range(len(neuron_count_array) - 1)]
        else: # sigma is a number - random init N(0,sigma)
            W = [sigma * np.random.randn(neuron_count_array[i + 1], neuron_count_array[i]) for i in range(len(neuron_count_array) - 1)]
            b = [0 * np.random.randn(neuron_count_array[i + 1], 1) for i in range(len(neuron_count_array) - 1)]
        return W, b


    def predict(self, X):
        Z, A = self.forwardpropagation(X, self.W, self.b)
        return np.reshape(np.argmax(A[-1], axis=0), (-1, 1))

    def forwardpropagation(self, X, W, B):
        Z = []
        if self.conv:
            X_afterconv = np.zeros((X.shape[2], 6272))
            for i in range(X.shape[2]):
                X_afterconv[i] = self.forwardpropagation_conv(X[:, :, i])
            A = [X_afterconv.T]
            #print("Conv ok")
        else:
            A = [X]
        for w, b in list(zip(W, B)):
            Z.append(np.dot(w, A[-1]) + b)
            if len(A) == len(W):
                A.append(softmax(Z[-1]))
            else:
                A.append(self.activation_function(Z[-1]))
        return np.asarray(Z), A

    def activation_error(self, Y, Z, A, W):
        delta_A = [0.] * len(self.W)
        y_error = np.transpose([np.bincount([y], minlength=np.shape(A[-1])[0]) for y in Y[0]])
        delta_A[-1] = -(y_error - A[-1]) * softmax_prim(Z[-1])

        for i in reversed(range(len(delta_A) - 1)):
            delta_A[i] = W[i + 1].T.dot(delta_A[i + 1]) * self.activation_function_prim(Z[i])
        return delta_A

    def backpropagation(self, y, Z, A, W):
        delta_A = self.activation_error(y, Z, A, W)
        batch_size = y.shape[1]
        delta_b = [np.sum(delta_a) / batch_size for delta_a in delta_A]
        delta_W = [np.dot(delta_a, a.T) / batch_size for delta_a, a in zip(delta_A, A)]
        self.deltaA3 = delta_A[0]
        return delta_W, delta_b

    def gradient_descent(self, X, Y, W, B, alpha):
        if self.alpha_optimizer != 'Nesterov' or self.previous_delta_W is None:
            (Z, A) = self.forwardpropagation(X, W, B)
            delta_W, delta_b = self.backpropagation(Y, Z, A, W)
        else:
            (Z, A) = self.forwardpropagation(X, W - 0.9 * np.array(self.previous_delta_W), B - 0.9 * np.array(self.previous_delta_b))
            delta_W, delta_b = self.backpropagation(Y, Z, A, W - 0.9 * np.array(self.previous_delta_W))
        #init optimization params
        if self.squared_deltas_W is None:
            self.squared_deltas_W = np.zeros_like(delta_W)
        if self.squared_deltas_b is None:
            self.squared_deltas_b = np.zeros_like(delta_b)
        if self.squared_previous_W is None:
            self.squared_previous_W = np.zeros_like(delta_W)
        if self.squared_previous_b is None:
            self.squared_previous_b = np.zeros_like(delta_b)
        if self.mW is None:
            self.mW = np.zeros_like(delta_W)
        if self.vW is None:
            self.vW = np.zeros_like(delta_W)
        if self.mb is None:
            self.mb = np.zeros_like(delta_b)
        if self.vb is None:
            self.vb = np.zeros_like(delta_b)
        if self.alpha_optimizer is None or self.previous_delta_W is None:
            self.W -= alpha * np.array(delta_W)
            self.b -= alpha * np.array(delta_b)
        elif self.alpha_optimizer == 'Momentum':
            self.W -= (0.9 * np.array(self.previous_delta_W) + alpha * np.array(delta_W))
            self.b -= (0.9 * np.array(self.previous_delta_b) + alpha * np.array(delta_b))
        elif self.alpha_optimizer == 'Nesterov':
            #nesterov changes also are before this
            self.W -= (0.9 * np.array(self.previous_delta_W) + alpha * np.array(delta_W))
            self.b -= (0.9 * np.array(self.previous_delta_b) + alpha * np.array(delta_b))
        elif self.alpha_optimizer == 'Adagrad':
            for i in range(len(delta_W)):
                self.squared_deltas_W[i] += delta_W[i]**2
                self.W[i] -= alpha * np.array(delta_W)[i] / (np.sqrt(self.squared_deltas_W[i]) + 1e-8)
                self.squared_deltas_b[i] += delta_b[i] ** 2
                self.b[i] -= alpha * np.array(delta_b)[i] / (np.sqrt(self.squared_deltas_b[i]) + 1e-8)

        elif self.alpha_optimizer == 'Adadelta':
            for i in range(len(delta_W)):
                self.squared_deltas_W[i] = 0.9 * self.squared_deltas_W[i] + 0.1 * delta_W[i] ** 2
                rms_gW = np.sqrt(self.squared_deltas_W[i]) + 1e-8
                previous_W = 0.9 * self.squared_previous_W[i] + 0.1 * self.previous_delta_W[i] ** 2
                rms_thetaW = np.sqrt(previous_W) + 1e-8
                self.W[i] -= np.divide(rms_thetaW, rms_gW) * np.array(delta_W)[i]
                self.squared_previous_W[i] = previous_W
                self.squared_deltas_b[i] = 0.9 * self.squared_deltas_b[i] + 0.1 * delta_b[i] ** 2
                rms_gb = np.sqrt(self.squared_deltas_b[i]) + 1e-8
                previous_b = 0.9 * self.squared_previous_b[i] + 0.1 * self.previous_delta_b[i] ** 2
                rms_thetab = np.sqrt(previous_b) + 1e-8
                self.b[i] -= np.divide(rms_thetab, rms_gb) * np.array(delta_b)[i]
                self.squared_previous_b[i] = previous_b
        elif self.alpha_optimizer == 'Adam':
            for i in range(len(delta_W)):
                self.mW[i] = 0.9 * self.mW[i] + (1. - 0.9) * np.array(delta_W)[i]
                self.vW[i] = 0.999 * self.vW[i] + (1. - 0.999) * np.array(delta_W)[i] ** 2
                mW_better = np.divide(self.mW[i], (1. - 0.9))
                vW_better = np.divide(self.vW[i], (1. - 0.999))
                self.W[i] -= alpha * mW_better / (np.sqrt(vW_better) + 1e-8)
                self.mb[i] = 0.9 * self.mb[i] + (1. - 0.9) * np.array(delta_b)[i]
                self.vb[i] = 0.999 * self.vb[i] + (1. - 0.999) * np.array(delta_b)[i] ** 2
                mb_better = np.divide(self.mb[i], (1. - 0.9))
                vb_better = np.divide(self.vb[i], (1. - 0.999))
                self.b[i] -= alpha * mb_better / (np.sqrt(vb_better) + 1e-8)
        else:
            raise Exception("Wrong alpha optimizer!")
        self.previous_delta_W = delta_W
        self.previous_delta_b = delta_b

        if self.conv:
            delta_W1_all = np.zeros(self.filterW.shape)
            delta_W3_reshaped = delta_W[0].reshape((self.deltaA3.shape[0], np.prod(self.before_flatten_shape))).T
            for i in range(X.shape[2]):
                delta_W3 = delta_W3_reshaped.dot(self.deltaA3[:,i])
                delta_W2 = self.backprop_max_pooling(delta_W3.reshape(self.before_flatten_shape))
                delta_W1 = self.backprop_conv(delta_W2)
                delta_W1_all += delta_W1
            self.filterW -= (alpha / X.shape[2]) * delta_W1_all
            #print("Conv backprop ok")



    def learn(self, X_train, Y_train, X_val, Y_val, max_epochs, alpha, batch_size):

        batch_size = min(batch_size, X_train.shape[2 if self.conv else 1])
        batch_count = X_train.shape[2 if self.conv else 1] // batch_size
        epoch = 0

        while epoch < max_epochs:
            epoch += 1
            print(epoch)
            for i in range(batch_count):
                sub = np.random.randint(X_train.shape[2 if self.conv else 1], size=batch_size)
                X_batch, Y_batch = (X_train[:, :, sub] if self.conv else X_train[:, sub]), Y_train[:, sub]
                self.gradient_descent(X_batch, Y_batch, self.W, self.b, alpha)
            train_accuracy = self.test_correctness(X_train, Y_train)
            print(train_accuracy)
            val_accuracy = self.test_correctness(X_val, Y_val)
            #print(val_accuracy)
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.best_W = self.W
                self.best_b = self.b
            if train_accuracy > 0.99:
                print ("99% acc reached at epoch: " + str(epoch))
                self.W = self.best_W
                self.b = self.best_b
                return
        self.W = self.best_W
        self.b = self.best_b

    def test_correctness(self, X, Y):
        y = np.reshape(Y, (-1))
        pred = np.reshape(self.predict(X), (-1))
        return np.count_nonzero(pred == y) / len(y)

    def forwardpropagation_conv(self, X):
        X_after_conv1 = self.forward_firstconvlayer(X)
        X_after_maxpooling = self.max_pooling(relu(X_after_conv1))
        return X_after_maxpooling.flatten()

    def forward_firstconvlayer(self, X):
        self.previous_conv_X = X
        result = np.zeros((X.shape[0] - (self.filter_size - 2) * 2, X.shape[1] - (self.filter_size - 2) * 2, self.filter_count))
        for region, i, j in self.regions_firstconvlayer(X):
            result[i, j] = np.sum(region * self.filterW, axis=(1, 2))
        return result

    def max_pooling(self, X):
        self.previous_max_pooling_X = X
        result = np.zeros((X.shape[0] // 2, X.shape[1] // 2, X.shape[2]))
        self.before_flatten_shape = (X.shape[0] // 2, X.shape[1] // 2, X.shape[2])
        for region, i, j in self.regions_maxpooling(X):
            result[i, j] = np.amax(region, axis=(0, 1))

        return result

    def backprop_max_pooling(self, delta_W3):
        delta_W2 = np.zeros(self.previous_max_pooling_X.shape)

        for region, i, j in self.regions_maxpooling(self.previous_max_pooling_X):
            height, width , filtercount = region.shape
            maxvals = np.amax(region, axis=(0, 1))

            for h in range(height):
                #for w in range(width):
                    for f in range(filtercount):
                        if region[h, h, f] == maxvals[f]:
                            delta_W2[i * 2 + h, j * 2 + h, f] = delta_W3[i, j, f]

        return delta_W2

    def backprop_conv(self, delta_W2):
        delta_W1 = np.zeros(self.filterW.shape)

        for x_region, i, j in self.regions_firstconvlayer(self.previous_conv_X):
            for f in range(self.filter_count):
                delta_W1[f] += delta_W2[i, j, f] * x_region
        return delta_W1

    def regions_firstconvlayer(self, X):
        for i in range(X.shape[0] - (self.filter_size - 2) * 2):
            for j in range(X.shape[1] - (self.filter_size - 2) * 2):
                region = X[i:(i + self.filter_size), j:(j + self.filter_size)]
                yield region, i, j

    def regions_maxpooling(self, X):
        #max-pooling 2x2
        for i in range(X.shape[0] // 2):
            for j in range(X.shape[1] // 2):
                region = X[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield region, i, j
