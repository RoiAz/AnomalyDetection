from KitNET.expriement import hyperparms
from KitNET.utils import *
import torch
import torch.nn as nn
import torch.optim as optim

hp = hyperparms()
rng = numpy.random.RandomState(1234)


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        modules = []
        channel_list = [in_channels] + [2] + [4] + [out_channels]
        for ci in range(1, len(channel_list)):
            modules.append(
                nn.Conv1d(in_channels=channel_list[ci - 1], out_channels=channel_list[ci], kernel_size=3, stride=2))
            modules.append(nn.BatchNorm1d(channel_list[ci]))
            modules.append(nn.LeakyReLU(negative_slope=0.05))
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        print(x.shape)
        print(x)
        x = torch.tensor(x)
        print(x.shape)
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        modules = []
        channel_list = [in_channels] + [4] + [2] + [out_channels]
        for ci in range(1, len(channel_list)):
            if ci == len(channel_list) - 1:
                modules.append(
                    nn.ConvTranspose1d(in_channels=channel_list[ci - 1], out_channels=channel_list[ci], kernel_size=3,
                                       stride=2, output_padding=1))
            else:
                modules.append(
                    nn.ConvTranspose1d(in_channels=channel_list[ci - 1], out_channels=channel_list[ci], kernel_size=3,
                                       stride=2))
            modules.append(nn.BatchNorm1d(channel_list[ci]))
            modules.append(nn.LeakyReLU(negative_slope=0.05))
        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        h = torch.tensor(h)
        return torch.tanh(self.cnn(h))


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


def create_optimizer(model_params, opt_params):
    opt_params = opt_params.copy()
    optimizer_type = opt_params['type']
    opt_params.pop('type')
    return optim.__dict__[optimizer_type](model_params, **opt_params)



class AutoEncoder(nn.Module):
    def __init__(self, loss_fn, n_visible=5, n_hidden=3):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = hp["lr"]
        self.input = 0
        self.L_h2 = 0
        self.encode_output = 0
        self.decode_output = 0
        self.encoder = None
        self.decoder = None
        self.loss_fn = loss_fn
        self.Net = hp["net"]
        self.min_kernal_size = 3
        if n_visible < 3:
            self.Net = "Kitsune"
            print("using Kitsune as default")


        if self.Net == "Kitsune":
            a = 1. / self.n_visible
            self.hbias = numpy.zeros(self.n_hidden)  # initialize h bias 0
            self.vbias = numpy.zeros(self.n_visible)  # initialize v bias 0
            self.W = numpy.array(rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(self.n_visible, self.n_hidden)))
            self.W_prime = self.W.T

        if self.Net == "PNet":
            self.in_channels = hp["in_channels"]
            self.out_channels = hp["out_channels"]
            self.encoder = EncoderCNN(in_channels=self.in_channels, out_channels=self.out_channels).to(device)
            print(self.encoder)
            self.decoder = DecoderCNN(in_channels=self.out_channels, out_channels=self.in_channels).to(device)
            print(self.decoder)
            if hp["opt"] == "adam":
                self.enc_optimizer = create_optimizer(self.encoder.parameters(), hp['enc_optimizer'])
                self.dec_optimizer = create_optimizer(self.decoder.parameters(), hp['dec_optimizer'])

    def encode(self, x):
        self.input = x
        if self.Net == "Kitsune":
            self.encode_output = sigmoid(numpy.dot(x, self.W) + self.hbias)
            print(self.encode_output)
            return self.encode_output
        elif self.Net == "PNet":
            return self.encoder.forward(x)

    def decode(self, x):
        if self.Net == "Kitsune":
            self.decode_output = sigmoid(numpy.dot(x, self.W_prime) + self.vbias)
            return self.decode_output
        elif hp["net"] == "PNet":
            return self.decoder.forward(x)

    def train(self):
        if self.Net == "Kitsune":
            self.L_h2 = self.input - self.decode_output
            L_h1 = numpy.dot(self.L_h2, self.W) * self.encode_output * (1 - self.encode_output)
            L_hbias = L_h1
            L_vbias = self.L_h2
            L_W = numpy.outer(self.input.T, L_h1) + numpy.outer(self.L_h2.T, self.encode_output)
            self.W += self.lr * L_W
            self.hbias += self.lr * L_hbias
            self.vbias += self.lr * L_vbias
            if hp["decode_train"]:
                self.W_prime += self.lr * L_W.T
        else:
            self.enc_optimizer.zero_grad()
            self.dec_optimizer.zero_grad()
            output = self.decode_output
            loss = self.loss_fn(output, self.input)
            loss.backward()
            self.optimizer.step()

    def calculateError(self):
        return numpy.sqrt(numpy.mean(self.L_h2 ** 2))


class Loss:
    def __init__(self):
        if hp["loss"] == "Krmse":
            self.loss_fn = lambda x, z: numpy.sqrt(((x - z) ** 2).mean())

        if hp["loss"] == "CE":
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y=0):
        return self.loss_fn(x, y)
