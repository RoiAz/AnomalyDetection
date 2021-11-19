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
        # channel_list = [in_channels] + [2] + [4] + [out_channels]
        channel_list = [in_channels] + [out_channels]
        for ci in range(1, len(channel_list)):
            modules.append(
                nn.Conv1d(in_channels=channel_list[ci - 1], out_channels=channel_list[ci], kernel_size=2, stride=1,
                          dtype=torch.double))
            modules.append(nn.BatchNorm1d(channel_list[ci], dtype=torch.double))
            modules.append(nn.LeakyReLU(negative_slope=0.05))
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.double, requires_grad=True)
        x = torch.reshape(x, (-1,))
        x = torch.reshape(x, (1, 1, x.shape[0]))
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        modules = []
        # channel_list = [in_channels] + [4] + [2] + [out_channels]
        channel_list = [in_channels] + [out_channels]
        for ci in range(1, len(channel_list)):
            if ci == len(channel_list) - 1:
                modules.append(
                    nn.ConvTranspose1d(in_channels=channel_list[ci - 1], out_channels=channel_list[ci], kernel_size=2,
                                       stride=1, dtype=torch.double))
            else:
                modules.append(
                    nn.ConvTranspose1d(in_channels=channel_list[ci - 1], out_channels=channel_list[ci], kernel_size=2,
                                       stride=1, dtype=torch.double))
            modules.append(nn.BatchNorm1d(channel_list[ci], dtype=torch.double))
        #  modules.append(nn.LeakyReLU(negative_slope=0.05))
        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        h = torch.tanh(self.cnn(h))
        h = torch.reshape(h, (-1,))
        h = torch.reshape(h, (1, h.shape[0]))
        return h


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
    def __init__(self, n_visible=5, n_hidden=3):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = hp["lr"]
        self.input = 0
        self.L_h2 = 0
        self.encode_output = 0
        self.decode_output = 0
        self.encoder = None
        self.decoder = None
        self.loss_fn = None
        self.err_fn = None
        self.Net = hp["net"]
        self.min_kernal_size = 3
        self.default = False
        if n_visible < self.min_kernal_size:
            self.default = True
            self.Net = "Kitsune"
            self.loss_fn = Loss("Krmse")
            print("using Kitsune as default")
        else:
            self.loss_fn = Loss(hp["loss"])
            print("using: " + self.Net)
        if self.Net == "Kitsune":
            a = 1. / self.n_visible
            self.hbias = numpy.zeros(self.n_hidden)  # initialize h bias 0
            self.vbias = numpy.zeros(self.n_visible)  # initialize v bias 0
            self.W = numpy.array(rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(self.n_visible, self.n_hidden)))
            self.W_prime = self.W.T
            self.err_fn = lambda x, z: numpy.sqrt(numpy.mean((x - z) ** 2))

        else:
            if hp["err_func"] == "rmse":
                criterion = nn.MSELoss()
                self.err_fn = lambda x, z: torch.sqrt(criterion(x, z))
            self.in_channels = hp["in_channels"]
            self.out_channels = hp["out_channels"]
            self.encoder = EncoderCNN(in_channels=self.in_channels, out_channels=self.out_channels).to(device)
            #      print(self.encoder)
            self.decoder = DecoderCNN(in_channels=self.out_channels, out_channels=self.in_channels).to(device)
            #     print(self.decoder)
            self.z_dim = hp['z_dim']
            self.features_shape, n_features = self._check_features(n_visible)
            self.mu_a = nn.Linear(n_features, self.z_dim, bias=True,dtype=torch.double)
            self.log_sigma = nn.Linear(n_features, self.z_dim, bias=True,dtype=torch.double)
            self.x_to_h = nn.Linear(self.z_dim, n_features, bias=True,dtype=torch.double)
            if hp["opt"] == "adam":
                self.optimizer = optim.Adam(self.parameters(), lr=self.lr, betas=hp["betas"])
        #     print(self.optimizer)

    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            x = torch.randn(1, in_size, device=device, dtype=torch.double)
            h = self.encoder(x)
            xr = self.decoder(h)
            #  print(x.shape)
            # print(xr.shape)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h) // h.shape[0]

    def encode(self, x):
     #   print(x)
        if self.Net == "Kitsune":
            self.input = x
            self.encode_output = sigmoid(numpy.dot(x, self.W) + self.hbias)
            return self.encode_output
        # elif self.Net == "PNet":
        else:
            x = torch.tensor(x, dtype=torch.double, requires_grad=True)
            self.input = torch.reshape(x, (-1,))
            init_z = self.encoder.forward(x)
            init_z = init_z.view(init_z.size(0), -1)
            mu = self.mu_a(init_z)
            log_sigma2 = self.log_sigma(init_z)
            u = torch.randn_like(mu)
            sig = torch.exp(log_sigma2)
            z = mu + u * sig
            self.encode_output = z
        return self.encode_output

    def decode(self, x):
        if self.Net == "Kitsune":
            self.decode_output = sigmoid(numpy.dot(x, self.W_prime) + self.vbias)
        # elif self.Net == "PNet":
        else:
            h = self.x_to_h(x)
            h = h.reshape((-1, *self.features_shape))
            x_rec = self.decoder.forward(h)
            x_rec = torch.tanh(x_rec)
            self.decode_output = torch.reshape(x_rec, (-1,))
        return self.decode_output

    def execute(self, x):
        if self.Net != "Kitsune":
            x = torch.tensor(x, dtype=torch.double)
        h = self.encode(x)
        x_rec = self.decode(h)
        loss = self.loss_fn.forward(x, x_rec)
        if self.Net != "Kitsune":
            loss = loss.detach().numpy()
        return loss

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
            self.optimizer.zero_grad()
            output = self.decode_output
            # print("%" * 10)
            # print(self.input.shape)
            # print(output.shape)
            loss = self.loss_fn.forward(output, self.input)
            loss.backward()
            self.optimizer.step()

    def calculateError(self):
        return self.err_fn(self.input, self.decode_output)


class Loss:
    def __init__(self, exp_loss):
        self.exp_loss = exp_loss

        if exp_loss == "Krmse":
            self.loss_fn = lambda x, z: numpy.sqrt(((x - z) ** 2).mean())
        elif exp_loss == "L1":
            self.loss_fn = nn.L1Loss()
        elif exp_loss == "MSE":
            self.loss_fn = nn.MSELoss()
        elif exp_loss == "CE":
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            print("No such loss function: " + exp_loss)
            exit()

    def forward(self, x, y):
        if self.exp_loss == "CE":
            return self.loss_fn(torch.reshape(x, (len(x), 1)), y)
        else:
            return self.loss_fn(x, y)
