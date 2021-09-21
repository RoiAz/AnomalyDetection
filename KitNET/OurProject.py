from expriement import hyperparms
from KitNET.utils import *

hp = hyperparms()


class Encoder:
    def __init__(self,):
        pass

    def encode(self, input):
        if hp["encoder"] == "Kitsune":
            return sigmoid(numpy.dot(input, self.W) + self.hbias)
        elif hp["encoder"] == "HerskoNet":
            pass



