import torch
from torch import nn
import torch.nn.functional as F
from model.util import fanin_init


# actor/critic parent class
class ActorCriticModule(nn.Module):
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def set_parameters(self, source):
        for target_param, param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def update_parameters(self, source, learning_rate):
        for target_param, param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_((1 - learning_rate) * target_param + learning_rate * param)


# policy nn
class Actor(ActorCriticModule):
    def __init__(self, num_states, num_actions, hidden1=400, hidden2=300, init_w=3e-3, parameters_source=None):
        super(Actor, self).__init__()

        self.num_states = num_states
        self.num_actions = num_actions

        self.fc1 = nn.Linear(num_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        if parameters_source is None:
            self.init_weights(init_w)
        else:
            self.set_parameters(parameters_source)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return F.softmax(out, dim=0)


# critic nn
class Critic(ActorCriticModule):
    def __init__(self, num_states, num_actions, hidden1=400, hidden2=300, init_w=3e-3, parameters_source=None):
        super(Critic, self).__init__()

        self.num_states = num_states
        self.num_actions = num_actions

        self.fc1 = nn.Linear(num_states, hidden1)
        self.fc2 = nn.Linear(hidden1 + num_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        if parameters_source is None:
            self.init_weights(init_w)
        else:
            self.set_parameters(parameters_source)

    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = self.relu(out)

        out = self.fc2(torch.cat([out, a], 1))
        out = self.relu(out)
        out = self.fc3(out)
        return out


