from KitNET.expriement import hyperparms
from KitNET.utils import *

hp = hyperparms()
rng = numpy.random.RandomState(1234)


class Norm:
    def __init__(self, n_visible):
        self.n_visible = n_visible
        if hp["norm"] == "Knorm":
            # for 0-1 normlaization
            self.norm_max = numpy.ones((self.n_visible,)) * -numpy.Inf
            self.norm_min = numpy.ones((self.n_visible,)) * numpy.Inf

    def norm(self, x, train_mode=True):
        if hp["norm"] == "Knorm":
            if train_mode:
                # update norms
                self.norm_max[x > self.norm_max] = x[x > self.norm_max]
                self.norm_min[x < self.norm_min] = x[x < self.norm_min]
            # 0-1 normalize
            return (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)


class autoEncoder:
    def __init__(self, n_visible=5, n_hidden=3):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.elr = hp["encoder_lr"]
        self.dlr = hp["decoder_lr"]
        self.input = 0
        self.encode_output = 0
        self.decode_output = 0
        if hp["encoder"] == "Kitsune":
            a = 1. / self.n_visible
            self.hbias = numpy.zeros(self.n_hidden)  # initialize h bias 0
            self.vbias = numpy.zeros(self.n_visible)  # initialize v bias 0
            self.W = numpy.array(rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(self.n_visible, self.n_hidden)))
            self.W_prime = self.W.T

    def encode(self, x):
        self.input = x
        if hp["encoder"] == "Kitsune":
            self.encode_output = sigmoid(numpy.dot(x, self.W) + self.hbias)
            return self.encode_output
        elif hp["encoder"] == "PNet":
            pass

    def decode(self, x):
        if hp["decoder"] == "Kitsune":
            self.decode_output = sigmoid(numpy.dot(x, self.W_prime) + self.vbias)
            return self.decode_output
        elif hp["decoder"] == "PNet":
            pass

    def train(self):
        if hp["encoder"] == "Kitsune":
            self.L_h2 = self.input - self.decode_output
            L_h1 = numpy.dot(self.L_h2, self.W) * self.encode_output * (1 - self.encode_output)
            L_hbias = L_h1
            L_vbias = self.L_h2
            L_W = numpy.outer(self.input.T, L_h1) + numpy.outer(self.L_h2.T, self.encode_output)
            self.W += self.elr * L_W
            self.hbias += self.elr * L_hbias
            self.vbias += self.dlr * L_vbias
            if hp["decode_train"]:
                self.W_prime += self.dlr * L_W.T

    def calculateError(self):
        return numpy.sqrt(numpy.mean(self.L_h2 ** 2))
