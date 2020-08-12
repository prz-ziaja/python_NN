import numpy as np

np.random.seed(100)


class Layer():
    def __init__(self, input, output, acti='tanh'):
        self.input = input
        self.output = output
        self.weights = np.random.randn(output, input).astype('f') * np.sqrt(1 / input)
        self.bias = np.random.randn(output, 1).astype('f')
        self.inputs = None
        self.mid = None
        self.outputs = None
        self.activataion = {'tanh': lambda x: np.tanh(x),
                            'sigmoid': lambda x: np.divide(1, np.add(1, np.exp(-x))),
                            'relu': lambda x: np.maximum(0, x),
                            'leaky relu': lambda x: np.maximum(np.multiply(0.01, x), x)
                            }[acti]
        self.derivative = {'tanh': lambda x: np.subtract(1, np.power(x, 2)),
                           'sigmoid': lambda x: np.divide(np.exp(-x), np.power(1 + np.exp(-x), 2)),
                           'relu': lambda x: np.greater(x, 0).astype('f'),
                           'leaky relu': lambda x: np.add(np.greater(x, 0).astype('f'),
                                                          np.multiply(0.01, np.less(x, 0).astype('f')))
                           }[acti]

    def forward(self, inp):
        self.inputs = np.copy(inp)
        z = np.add(np.dot(self.weights, inp), self.bias)
        self.mid = np.copy(z)
        a = self.activataion(z)
        self.outputs = np.copy(a)
        return a

    def backward(self, w1, dz1):
        dz = np.multiply(np.dot(w1.T, dz1), self.derivative(self.mid))
        dw = np.dot(dz, self.inputs.T)
        dw = np.divide(dw, dz1.shape[1])
        return dw, dz

    def backwardst(self, y):
        da = (1 - y) / (1 - self.outputs) - y / self.outputs
        dz = da * self.derivative(self.outputs)
        # dz=np.subtract(self.outputs,y)
        dw = np.dot(dz, self.inputs.T)
        dw = np.divide(dw, y.shape[1])
        return dw, dz
