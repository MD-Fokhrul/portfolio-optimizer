import random
from collections import defaultdict
import numpy as np


class ReplayBuffer():
    def __init__(self, minibatch_size):
        self.minibatch_size = minibatch_size
        self.buffer_map = defaultdict(list)

    # store (s0, a0, r0, s1) in buffer
    def store(self, state, action, reward, result_state):
        self.buffer_map['state'].append(state)
        self.buffer_map['action'].append(action)
        self.buffer_map['reward'].append(reward)
        self.buffer_map['result_state'].append(result_state)

    # minibatch buffer sampling
    # returns minibatch sample for each key in buffer
    def get_sample_arrays_map(self):
        sample_buffer_map = {}
        sample_idxs = self.sample_buffer_idxs()

        for key in self.buffer_map.keys():
            sample_list = self.pick_sample(key, sample_idxs)
            sample_buffer_map[key] = np.vstack(sample_list)

        return sample_buffer_map

    # get buffer row idxs for minibatch sample
    # used by get_sample_arrays_map
    def sample_buffer_idxs(self):
        # TODO: is it ok to pick idx=0?
        buffer_len = len(self.buffer_map['state']) - 1
        full_idx_range = range(buffer_len)

        if buffer_len >= self.minibatch_size:
            batch_idxs = random.sample(full_idx_range, self.minibatch_size)
        else:
            # if we don't have enought examples yet we should sample with repeating idxs | can avoid this with warmup phase(?)
            batch_idxs = np.random.randint(0, max(buffer_len, 1), size=self.minibatch_size)

        # TODO: do we need to make sure we sample doesn't span across multiple episdes?
        return batch_idxs

    # select from list based on idxs
    # used by get_sample_arrays_map
    def pick_sample(self, key, idxs):
        sample_space = []
        for idx in idxs:
            sample_space.append(self.buffer_map[key][idx])

        return sample_space



# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
# Originally from https://github.com/edolev89/pytorch-ddpg/blob/master/random_process.py
class OrnsteinUhlenbeckProcess():
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.n_steps = 0
        self.reset_states()

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma

    def sample(self):
        # do we need epsilon/decay?
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(
            self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)


