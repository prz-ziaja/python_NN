# https://towardsdatascience.com/a-visual-explanation-of-gradient-descent-methods-momentum-adagrad-rmsprop-adam-f898b102325c
#
from Layer import Layer
import numpy as np


class Net():
    def __init__(self):
        self.layers = []
        self.batch_size = 40
        self.epochs = 5
        self.lam = 0.035
        self.alpha = 0.000007
        self.pr_sum_gr = 0
        self.pr_sum_dz = 0
        self.pr_sum_gr_sq = 0
        self.pr_sum_dz_sq = 0
        self.beta1 = 0.1
        self.beta2 = 0.1
        self.alpha_decay=1
    def add(self, input, output, activation='tanh'):
        if len(self.layers) == 0:
            if input < 0 or output < 0 or activation not in ('relu', 'tanh', 'sigmoid', 'leaky relu', 'softmax'):
                return
            else:
                self.layers.append(Layer(input, output, activation))
        else:
            if input != self.layers[-1].output or output < 0 or activation not in (
            'relu', 'tanh', 'sigmoid', 'leaky relu', 'softmax'):
                return
            else:
                print('ok')
                self.layers.append(Layer(input, output, activation))

    def addo(self, output, activation='tanh'):
        if len(self.layers) == 0:
            return
        else:
            if output < 0 or activation not in ('relu', 'tanh', 'sigmoid', 'leaky relu', 'softmax'):
                return
            else:
                self.layers.append(Layer(self.layers[-1].output, output, activation))

    def predict(self, x):
        inp = self.layers[0].forward(x)
        for j in self.layers[1:]:
            inp = j.forward(inp)
        return inp

    def epoch(self, x, y):
        for i in range(0, x.shape[1], self.batch_size):
            self.predict(x[:, i:i + self.batch_size])
            grads = []
            grads.append(self.layers[-1].backwardst(y[:, i:i + self.batch_size]))
            for j in range(len(self.layers) - 2, -1, -1):
                grads.append(self.layers[j].backward(self.layers[j + 1].weights, grads[-1][1]))
            grads = list(reversed(grads))
            for i in range(len(self.layers)):
                # dw = self.alpha*(grads[i][0]+(self.lam/self.batch_size)*self.layers[i].weights)
                # self.prgrad[i] = (1-self.decay)*dw+self.decay*self.prgrad[i]
                gr = grads[i][0] + (self.lam / self.batch_size) * self.layers[i].weights
                self.pr_sum_gr[i] = self.pr_sum_gr[i] * self.beta1 + gr * (1 - self.beta1)
                self.pr_sum_gr_sq[i] = self.pr_sum_gr_sq[i] * self.beta2 + np.power(gr, 2) * (1 - self.beta2)
                self.layers[i].weights = self.layers[i].weights - self.alpha * self.pr_sum_gr[i] / self.pr_sum_gr_sq[i]
                # self.prdz[i] = (1-self.decay)*self.alpha*np.mean(grads[i][1], axis=1,keepdims=True)+self.decay*self.prdz[i]
                db = np.mean(grads[i][1], axis=1, keepdims=True)
                self.pr_sum_dz[i] = self.pr_sum_dz[i] * self.beta1 + db * (1 - self.beta1)
                self.pr_sum_dz_sq[i] = self.pr_sum_dz_sq[i] * self.beta2 + np.power(db, 2) * (1 - self.beta2)
                self.layers[i].bias = self.layers[i].bias - self.alpha * self.pr_sum_dz[i] / self.pr_sum_dz_sq[i]

    def cost(self, x, y):
        est = self.predict(x)
        cost = -np.mean(
            np.add(np.multiply(y, np.log(est)), np.multiply(np.subtract(1, y), np.log(np.subtract(1, est)))))
        return cost

    def fit(self, x, y):
        self.pr_sum_gr = [0 for i in range(len(self.layers))]
        self.pr_sum_dz = [0 for i in range(len(self.layers))]
        self.pr_sum_gr_sq = [0 for i in range(len(self.layers))]
        self.pr_sum_dz_sq = [0 for i in range(len(self.layers))]
        if x.shape[0] != self.layers[0].input or y.shape[0] != self.layers[-1].output:
            return
        alpha=self.alpha
        for i in range(self.epochs):
            print(i, ': ', self.cost(x, y))
            self.epoch(x, y)
            self.alpha*=self.alpha_decay
        self.alpha=alpha
