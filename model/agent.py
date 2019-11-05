import torch
from torch import nn
import numpy as np
from model.nn import Actor, Critic
from model.misc import ReplayBuffer, OrnsteinUhlenbeckProcess
from model.util import extract_tensors_from_buffer_map, to_cuda_if_needed, to_tensor


class DDPG():
    def __init__(self, num_states, num_actions, minibatch_size, random_process_args, device_type='cpu',
                 learning_rate=0.001, discount_factor=0.9, is_training=True, linear_action_epsilon_decay=50000):
        self.device_type = device_type
        self.device = torch.device(device_type)

        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.action_epsilon = 1.0
        self.decay_action_epsilon_delta = 1.0 / linear_action_epsilon_decay

        self.num_states = num_states
        self.num_actions = num_actions
        self.minibatch_size = minibatch_size
        self.is_training = is_training

        self.replay_buffer = ReplayBuffer(minibatch_size)

        self.actor = Actor(num_states, num_actions)  # mu
        self.critic = Critic(num_states, num_actions)  # Q

        self.actor_target = Actor(self.actor.num_states, self.actor.num_actions, parameters_source=self.actor)  # mu'
        self.critic_target = Critic(self.critic.num_states, self.critic.num_actions, parameters_source=self.critic)  # Q'

        to_cuda_if_needed([self.actor, self.critic, self.actor_target, self.critic_target], device_type)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.critic_criterion = nn.MSELoss()
        self.random_process = OrnsteinUhlenbeckProcess(size=num_actions, **random_process_args)

    def update_policy(self):
        minibatch_arrays_map = self.replay_buffer.get_sample_arrays_map()  # sample random minibatch from R (s(i), a(i), r(i), s(i+1))
        states_minibatch, actions_minibatch, rewards_minibatch, result_states_minibatch = extract_tensors_from_buffer_map(minibatch_arrays_map)

        next_q_minibatch = self.critic_target((result_states_minibatch, self.actor_target(result_states_minibatch)))  # should this be next_state or result_states_minibatch? paper has it one time as s(t+1) and one time as s(i+1). Guessing it's s(i+1)
        # do we need something about terminal states for target_q_minibatch?
        target_q_minibatch = rewards_minibatch + self.discount_factor * next_q_minibatch  # set y(i) = r(i) + γq' | where γ is the discount factor, q' is critic_target output for [s(t+1), B], and  is action by actor_target(s(i+1))

        self.critic.zero_grad()
        q_minibatch = self.critic((states_minibatch, actions_minibatch))
        value_loss = self.critic_criterion(target_q_minibatch, q_minibatch)  # does the order matter? paper says (targetq - q), other ddpg imps say loss(q, targetq)
        value_loss.backward()
        self.critic_optim.step()

        self.actor.zero_grad()
        policy_loss = (
            -self.critic((states_minibatch, self.actor(states_minibatch)))).mean()  # TODO: unsure about this
        policy_loss.backward()
        self.actor_optim.step()

        self.critic_target.update_parameters(self.critic, self.learning_rate)
        self.actor_target.update_parameters(self.actor, self.learning_rate)

    def append_observation(self, current_state, current_action, current_reward, next_state):
        self.replay_buffer.store(current_state, current_action, current_reward, next_state)

    # let actor score actions, add random noise and clip to make sure values still in -1,1 range
    # input: nx1 state array
    def select_action(self, state, decay_epsilon=True):
        state = to_tensor(state, self.device)
        noise = max(self.action_epsilon, 0) * self.action_noise()

        if decay_epsilon:
            self.action_epsilon -= self.decay_action_epsilon_delta

        # TODO: do we need to normalize after adding noise?
        action = self.actor(state)
        # for tens, name in [(action, 'action'), (noise, 'noise')]:
        #     arr = tens
        #     if name == 'action':
        #         arr = arr.detach().numpy()
        #     print('%s sum: %2f, low: %2f, high: %2f' % (name, np.sum(arr), np.min(arr), np.max(arr)))
        # return torch.clamp(action + noise, 0, 1.).detach()
        return np.clip((action).detach().numpy() + noise, 0, 1)

    def action_noise(self):
        sample = self.random_process.sample()
        return (sample / np.sum(sample)) / 2 - 0.25 if self.is_training else 0

    def select_random_action(self):
        random_action = np.random.uniform(0, 1., self.num_actions)
        return random_action / np.sum(random_action)

    def reset_action_noise_process(self):
        self.random_process.reset_states()






