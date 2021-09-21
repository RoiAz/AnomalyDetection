from expriement import hyperparms
from KitNET.utils import *

hp = hyperparms()


class Encoder:
    def __init__(self, n_visible=5, n_hidden=3):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        if hp["encoder"] == "Kitsune":
            self.W = numpy.array(self.rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(self.params.n_visible, self.params.n_hidden)))
        pass

    def encode(self, input):
        if hp["encoder"] == "Kitsune":
            return sigmoid(numpy.dot(input, self.W) + self.hbias)
        elif hp["encoder"] == "HerskoNet":
            pass



class decoder:
    def __init__(self,):
        pass

    def decode(self, input):
        if hp["decoder"] == "Kitsune":
            return sigmoid(numpy.dot(input, self.W) + self.hbias)
        elif hp["decoder"] == "HerskoNet":
            pass

