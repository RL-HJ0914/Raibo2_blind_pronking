import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal


class Actor:
    def __init__(self, architecture, distribution, device='cpu'):
        super(Actor, self).__init__()

        self.architecture = architecture
        self.distribution = distribution
        self.architecture.to(device)
        self.distribution.to(device)
        self.device = device
        self.action_mean = None

    def forward(self, obs):
        return self.architecture.forward(obs).squeeze(dim=0).cpu().numpy()

    def sample(self, obs):
        self.action_mean = self.architecture.forward(obs)
        self.action_mean = self.action_mean.squeeze(dim=0).cpu().numpy()
        actions, log_prob = self.distribution.sample(self.action_mean)
        return actions, log_prob

    def evaluate(self, obs, actions):
        self.action_mean = self.architecture.forward(obs)
        return self.distribution.evaluate(self.action_mean.view(-1, *self.action_shape), actions), self.action_mean

    def parameters(self):
        return [*self.architecture.parameters(), *self.distribution.parameters()]

    def save_deterministic_graph(self, file_name, example_input, device='cpu'):
        transferred_graph = torch.jit.trace(self.architecture.architecture.to(device), example_input)
        torch.jit.save(transferred_graph, file_name)
        self.architecture.architecture.to(self.device)

    def deterministic_parameters(self):
        return self.architecture.parameters()

    def update(self):
        self.distribution.update()

    @property
    def obs_shape(self):
        return self.architecture.input_shape

    @property
    def action_shape(self):
        return self.architecture.output_shape


class Critic:
    def __init__(self, architecture, device='cpu'):
        super(Critic, self).__init__()
        self.architecture = architecture
        self.architecture.to(device)

    def predict(self, obs):
        return self.architecture.architecture(obs).detach()

    def evaluate(self, obs):
        return self.architecture.architecture(obs).view(-1, 1)

    def parameters(self):
        return [*self.architecture.parameters()]

    @property
    def obs_shape(self):
        return self.architecture.input_shape


class MLP(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size):
        super(MLP, self).__init__()
        self.activation_fn = actionvation_fn

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))
        self.architecture = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.architecture, scale)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


class MultivariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, dim, size, init_std, fast_sampler, seed=0):
        super(MultivariateGaussianDiagonalCovariance, self).__init__()
        self.dim = dim
        self.std = nn.Parameter(init_std * torch.ones(dim))
        self.distribution = None
        self.fast_sampler = fast_sampler
        self.fast_sampler.seed(seed)
        self.samples = np.zeros([size, dim], dtype=np.float32)
        self.logprob = np.zeros(size, dtype=np.float32)
        self.std_np = self.std.detach().cpu().numpy()

    def update(self):
        self.std_np = self.std.detach().cpu().numpy()

    def sample(self, logits):
        self.fast_sampler.sample(logits, self.std_np, self.samples, self.logprob)
        return self.samples.copy(), self.logprob.copy()

    def evaluate(self, logits, outputs):
        distribution = Normal(logits, self.std.reshape(self.dim))
        actions_log_prob = distribution.log_prob(outputs).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)
        return actions_log_prob, entropy

    def entropy(self):
        return self.distribution.entropy()

    def enforce_minimum_std(self, min_std):
        current_std = self.std.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std.data = new_std

    def enforce_maximum_std(self, max_std):
        current_std = self.std.detach()
        new_std = torch.min(current_std, max_std.detach()).detach()
        self.std.data = new_std

    def enforce_std(self, std):
        self.std.data = std

class GRU_MLP_Actor(nn.Module):
    def __init__(self, measurement_dim, hidden_dim, mlp_shape, output_size, batch_size, device='cuda'):
        super(GRU_MLP_Actor, self).__init__()
        self.device = device
        self.measurement_dim = measurement_dim
        self.hidden_dim = hidden_dim
        self.mlp_shape = mlp_shape
        self.batch_size = batch_size
        self.GRU = nn.GRU(input_size=self.measurement_dim, hidden_size=self.hidden_dim, num_layers=1).to(self.device)
        self.MLP = MLP(shape=self.mlp_shape, actionvation_fn=nn.LeakyReLU, input_size=self.hidden_dim + self.measurement_dim + 3, output_size=output_size).to(self.device)
        self.h = torch.zeros([1, self.batch_size, self.hidden_dim], dtype=torch.float32).to(self.device)
        self.init_weights(self.GRU)
        self.input_shape = [measurement_dim + 3]
        self.output_shape = [output_size]

    def forward(self, input):
        latent, self.h = self.GRU(input[:, :, :self.measurement_dim], self.h)
        action = self.MLP.architecture(torch.cat((latent, input), dim=2))

        return action

    def init_hidden(self):
        self.h = torch.zeros([1, self.batch_size, self.hidden_dim], dtype=torch.float32).to(self.device)

    def init_by_done(self, arg_dones):
        self.h[:, arg_dones, :] = torch.zeros([1, arg_dones.size, self.hidden_dim], dtype=torch.float32).to(self.device)

    @staticmethod
    def init_weights(rnn):
        for layer in range(len(rnn.all_weights)):
            torch.nn.init.kaiming_normal_(rnn.all_weights[layer][0])
            torch.nn.init.kaiming_normal_(rnn.all_weights[layer][1])