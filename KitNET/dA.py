# Copyright (c) 2017 Yusuke Sugomori
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Portions of this code have been adapted from Yusuke Sugomori's code on GitHub: https://github.com/yusugomori/DeepLearning

import sys
import numpy
from KitNET.utils import *
import json
from KitNET.OurProject import *


class dA_params:
    def __init__(self, n_visible=5, n_hidden=3, lr=0.001, corruption_level=0.0, gracePeriod=10000, hiddenRatio=None):
        self.n_visible = n_visible  # num of units in visible (input) layer
        self.n_hidden = n_hidden  # num of units in hidden layer
        self.lr = lr
        self.corruption_level = corruption_level
        self.gracePeriod = gracePeriod
        self.hiddenRatio = hiddenRatio


class dA:
    def __init__(self, params):
        self.params = params
        if self.params.hiddenRatio is not None:
            self.params.n_hidden = int(numpy.ceil(self.params.n_visible * self.params.hiddenRatio))
        self.norm_func = Norm(self.params.n_visible)
        self.n = 0
        self.rng = numpy.random.RandomState(1234)
        self.autoencoder = AutoEncoder(self.params.n_visible, self.params.n_hidden)

    def get_corrupted_input(self, input, corruption_level):
        assert corruption_level < 1
        return self.rng.binomial(size=input.shape,
                                 n=1,
                                 p=1 - corruption_level) * input

    # Encode
    def get_hidden_values(self, input):
        return self.autoencoder.encode(input)

    # Decode
    def get_reconstructed_input(self, hidden):
        return self.autoencoder.decode(hidden)

    def train(self, x):
        self.n = self.n + 1
        x = self.norm_func.norm(x)
        if self.params.corruption_level > 0.0:
            tilde_x = self.get_corrupted_input(x, self.params.corruption_level)
        else:
            tilde_x = x
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        self.autoencoder.train()
        return self.autoencoder.calculateError()  # the RMSE reconstruction error during training

    def reconstruct(self, x):
        y = self.get_hidden_values(x)
        z = self.get_reconstructed_input(y)
        return z

    def execute(self, x):  # returns MSE of the reconstruction of x
        if self.n < self.params.gracePeriod:
            return 0.0
        else:
            x = self.norm_func.norm(x, train_mode=False)
            return self.autoencoder.execute(x)


    def inGrace(self):
        return self.n < self.params.gracePeriod
