import torch
from torch import nn
import numpy as np
from model.nn import Actor, Critic
from model.misc import ReplayBuffer, OrnsteinUhlenbeckProcess
from model.util import extract_tensors_from_buffer_map, to_cuda_if_needed, to_tensor, save_model, load_model


class DDPG():
    def __init__(self, num_states, num_actions, minibatch_size, random_process_args, device_type='cpu',
                 learning_rate=0.001, discount_factor=0.9, is_training=True, linear_action_epsilon_decay=50000):

        # cpu/cuda for tensors and modules
        self.device_type = device_type
        self.device = torch.device(device_type)

        # hyperparams
        self.discount_factor = discount_factor # discount factor for delayed rewards
        self.learning_rate = learning_rate # learning rate for nns
        self.action_epsilon = 1.0 # hyperparam for selected action random noise
        self.decay_action_epsilon_delta = 1.0 / linear_action_epsilon_decay # hyperparam for selected action random noise

        # sizes
        self.num_states = num_states # inputs to nns
        self.num_actions = num_actions # outputs of nns
        self.minibatch_size = minibatch_size # batch size
        self.is_training = is_training # training flag for gradients

        self.replay_buffer = ReplayBuffer(minibatch_size) # DDPG state memory buffer

        # nns for ddpg
        self.actor = Actor(num_states, num_actions)  # policy
        self.critic = Critic(num_states, num_actions)

        # target nns for ddpg
        self.actor_target = Actor(self.actor.num_states, self.actor.num_actions, parameters_source=self.actor)
        self.critic_target = Critic(self.critic.num_states, self.critic.num_actions, parameters_source=self.critic)

        to_cuda_if_needed([self.actor, self.critic, self.actor_target, self.critic_target], device_type)

        # optimizers, loss
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.critic_criterion = nn.MSELoss()

        # process to add decaying random noises to agent selected actions for better exploration
        self.random_process = OrnsteinUhlenbeckProcess(size=num_actions, **random_process_args)

    def update_policy(self):
        # sample random minibatch from R (s(i), a(i), r(i), s(i+1))
        minibatch_arrays_map = self.replay_buffer.get_sample_arrays_map()
        states_minibatch, actions_minibatch, rewards_minibatch, result_states_minibatch = extract_tensors_from_buffer_map(minibatch_arrays_map)

        # predict action with target policy and critique
        next_predicted_action_minibatch = self.actor_target(result_states_minibatch).detach()
        next_q_minibatch = self.critic_target((result_states_minibatch, next_predicted_action_minibatch)).detach()

        # expected target labels
        # TODO: should add flag to ignore terminal states for target_q_minibatch?
        target_q_minibatch = rewards_minibatch + self.discount_factor * next_q_minibatch  # set y(i) = r(i) + γq' | where γ is the discount factor, q' is critic_target output for [s(t+1), B], and  is action by actor_target(s(i+1))

        # optimize critic
        self.critic.zero_grad()
        q_minibatch = self.critic((states_minibatch, actions_minibatch))
        value_loss = self.critic_criterion(q_minibatch, target_q_minibatch)
        critic_loss_val = value_loss.item()
        value_loss.backward()
        self.critic_optim.step()

        # optimize policy
        self.actor.zero_grad()
        policy_loss = (
            -self.critic((states_minibatch, self.actor(states_minibatch)))).mean()  # TODO: unsure about this
        actor_loss_val = policy_loss.item()
        policy_loss.backward()
        self.actor_optim.step()

        # soft update target actor/critic weights from actor/critic
        self.critic_target.update_parameters(self.critic, self.learning_rate)
        self.actor_target.update_parameters(self.actor, self.learning_rate)

        return critic_loss_val, actor_loss_val

    # add new observation to agent memory
    def append_observation(self, current_state, current_action, current_reward, next_state):
        self.replay_buffer.store(current_state, current_action, current_reward, next_state)

    # let actor generate actions from state and add random noise
    # input: nx1 state array
    def select_action(self, state, decay_epsilon=True):
        state = to_tensor(state, self.device)
        noise = max(self.action_epsilon, 0) * self.action_noise() # generate random noise

        if decay_epsilon:
            self.action_epsilon -= self.decay_action_epsilon_delta

        action = self.actor(state) # have policy determine action

        # add noise to action and clip weights to help the agent explore more actions (if training)
        action_res = action.detach().cpu().numpy() + noise

        # TODO: better to normalize to 0-1 instead of clipping?
        # return np.clip(action_res, 0, 1)
        return action_res / np.sum(action_res)

    # get random noise for action from random_process
    def action_noise(self):
        sample = self.random_process.sample() # get random noise sample

        # if we're not training we don't want to explore so no noise required.
        # we want to keep the noise small relative to the small weights
        return (sample / np.sum(sample)) / 2 - 0.25 if self.is_training else 0

    # for warmup phase. select completely random action
    def select_random_action(self):
        random_action = np.random.uniform(0, 1., self.num_actions) # from normal distribution
        return random_action / np.sum(random_action) # normalize, weights should sum to 1

    # reset random process decay
    def reset_action_noise_process(self):
        self.random_process.reset_states()

    def save_model(self, dir_path, identifier):
        save_model(dir_path, self.actor, self.critic, self.replay_buffer, identifier=identifier)

    def load_model(self, model_dir_path):
        load_model(model_dir_path, self.actor, self.critic, self.replay_buffer)










