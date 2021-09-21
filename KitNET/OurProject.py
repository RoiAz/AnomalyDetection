from KitNET.expriement import hyperparms
from KitNET.utils import *

hp = hyperparms()


class Encoder:
    def __init__(self, n_visible=5, n_hidden=3):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        if hp["encoder"] == "Kitsune":
            a = 1. / self.n_visible
            self.hbias = numpy.zeros(self.n_hidden)  # initialize h bias 0
            self.vbias = numpy.zeros(self.n_visible)  # initialize v bias 0
            self.rng = numpy.random.RandomState(1234)
            self.W = numpy.array(self.rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(self.n_visible, self.n_hidden)))

    def encode(self, x):
        if hp["encoder"] == "Kitsune":
            return sigmoid(numpy.dot(x, self.W) + self.hbias)
        elif hp["encoder"] == "HerskoNet":
            pass


class decoder:
    def __init__(self, ):
        pass

    def decode(self, input):
        if hp["decoder"] == "Kitsune":
            return sigmoid(numpy.dot(input, self.W) + self.hbias)
        elif hp["decoder"] == "HerskoNet":
            pass
