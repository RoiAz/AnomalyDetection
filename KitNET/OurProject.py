from KitNET.expriement import hyperparms
from KitNET.utils import *

hp = hyperparms()
rng = numpy.random.RandomState(1234)


def get_corrupted_input( input, corruption_level):
    assert corruption_level < 1

    return rng.binomial(size=input.shape,
                             n=1,
                             p=1 - corruption_level) * input


class Encoder:
    def __init__(self, n_visible=5, n_hidden=3):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = hp["encoder_lr"]
        if hp["encoder"] == "Kitsune":
            a = 1. / self.n_visible
            self.hbias = numpy.zeros(self.n_hidden)  # initialize h bias 0
            self.W = numpy.array(rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(self.n_visible, self.n_hidden)))

    def encode(self, x):
        if hp["encoder"] == "Kitsune":
            return sigmoid(numpy.dot(x, self.W) + self.hbias)
        elif hp["encoder"] == "HerskoNet":
            pass


    def train(self):



class decoder:
    def __init__(self, ):
        pass

    def decode(self, input):
        if hp["decoder"] == "Kitsune":
            return sigmoid(numpy.dot(input, self.W) + self.hbias)
        elif hp["decoder"] == "HerskoNet":
            pass
