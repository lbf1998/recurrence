import os
import random
import torch
import torch.nn
import numpy as np
import yaml
# 李茜茜
from model.iqn_net import iqn
from model.memory import LazyPrioritizedMultiStepMemory, LazyMultiStepMemory
from model.replay_buffer import ReplayBuffer
from utils.util import LinearSchedule
from utils.utils import evaluate_quantile_at_action, \
    calculate_quantile_huber_loss, update_params, preprocess_state

frame_history_len = 4


class ndqfn(object):
    def __init__(self, env, logdir, savedir):
        # self.state_dim = state_dim
        with open('../agent/iqn.yaml') as f:
            self.config = yaml.load(f, Loader=yaml.SafeLoader)
        self.action_dim = env.action_space.n
        in_channels = env.observation_space.shape[0]
        self.learning_rate = self.config['lr']
        self.memory_size = self.config['memory_size']
        self.USE_GPU = self.config['use_gpu']
        self.log_dir = logdir
        self.save_dir = savedir
        self.TARGET_REPLACE_ITER = self.config['update_interval']
        self.BATCH_SIZE = self.config['batch_size']
        self.GAMMA = self.config['gamma']
        self.epsilon_schedule = LinearSchedule(schedule_timesteps=4000, final_p=0.05,
                                               initial_p=1.0)
        self.epsilon = self.config['epsilon_train']
        self._current_time_step = 0
        self._begin_train = self.config['random_starts']
        self.best_eval_score = -np.inf

        self.pred_net = iqn(in_channels, 7*7*64, self.action_dim,
                            num_cosines=self.config['num_cosines'])
        self.target_net = iqn(in_channels, 7*7*64, self.action_dim,
                              num_cosines=self.config['num_cosines'])

        # self.rnd_net = RND(self.state_dim, self.action_dim)

        # simulator step counter
        self.memory_counter = 0
        # target network step counter
        self.learn_step_counter = 0

        # iqnconfig
        self.batch_size = self.config['batch_size']
        self.K = self.config['K']
        self.N = self.config['N']
        self.N_dash = self.config['N_dash']
        self.kappa = self.config['kappa']
        self.use_per = self.config['use_per']
        self.P = self.config['P']
        self.P_cosine_index = torch.linspace(self.config['P0'], self.config['Pn'], self.config['P'])

        # ceate the replay buffer
        if self.config['use_per']:
            beta_steps = (self.config['num_steps'] - self.config['start_steps']) / self.config['update_interval']
            self.memory = LazyPrioritizedMultiStepMemory(
                self.memory_size, env.observation_space.shape,
                self.config['device'], self.GAMMA, self.config['multi_step'], beta_steps=beta_steps)
        else:
            self.memory = LazyMultiStepMemory(
                self.memory_size, env.observation_space.shape,
                self.config['device'], self.GAMMA, self.config['multi_step'])

        # define optimizer
        self.optimizer = torch.optim.Adam(self.pred_net.parameters(), lr=self.learning_rate)

        # sync evac target
        self.update_target()

        # use gpu
        if self.USE_GPU:
            self.pred_net.to(self.config['device'])
            self.target_net.to(self.config['device'])
            # self.rnd_net.to(config.device)
            self.pred_net.cuda()
            self.target_net.cuda()
            self.P_cosine_index.cuda()

    def choose_action(self, state, epsilon=None):
        # x:state
        if epsilon is not None:
            epsilon_used = epsilon
        else:
            epsilon_used = self.epsilon
        # taus = torch.rand(self.config['K'])
        if random.random() < epsilon_used:
            action = np.random.randint(0, self.action_dim)
        else:
        # taus = get_optimistic(state)
            if self.USE_GPU:
                # taus = torch.tensor(taus).cuda()
                state = torch.tensor(state).cuda()
            q_out, _ = self.pred_net(state, taus)
            q_out = q_out.mean(dim=-2).squeeze()
            action = q_out.argmax().item()  # max action
            # action = torch.argmax(action_value, dim=1).data.cpu().numpy()

        return action

    def op_choose_action(self, state, epsilon=None):
        # x:state
        # if epsilon is not None:
        #     epsilon_used = epsilon
        # else:
        #     epsilon_used = self.epsilon
        # taus = torch.rand(config.K)
        # if random.random() < epsilon_used:
        #     action = np.random.randint(0, self.action_dim)
        # else:
        # state = preprocess_state(state)
        # with torch.no_grad():
        #     taus, norm_opt = get_optimistic(state)

        if self.USE_GPU:
            # taus = torch.tensor(taus).cuda()
            state = state.cuda()
        with torch.no_grad():
            q_out, norm_opt = self.pred_net(state)
            q_out = q_out.mean(dim=-2).squeeze()
            action = q_out.argmax().item()  # max action
        # action = torch.argmax(action_value, dim=1).data.cpu().numpy()

        return action, norm_opt

    def test_action(self, state):
        state = preprocess_state(state)

        # state = torch.FloatTensor(state).view((1, self.state_dim))
        tau = torch.rand(self.config['K'])
        if self.USE_GPU:
            state = torch.tensor(state).cuda()
            tau = tau.cuda()
        value, _ = self.pred_net(state, tau)
        action = value.mean(dim=-2).squeeze().argmax().detach().item()

        return action

    def positive_or_negative(self):
        if random.random() < 0.5:
            return 1
        else:
            return -1

    # def train_rnd(self, state, action):
    #     return self.rnd_net.update(state, action)

    # def batch_train_rnd(self, state_action):
    #     return self.rnd_net.batch_update(state_action)

    def computer_loss(self, tau, value, target_value):
        # * get the quantile huber loss
        u = target_value.unsqueeze(1) - value.unsqueeze(-1)
        huber_loss = 0.5 * u.abs().clamp(min=0., max=self.config['kappa']).pow(2)
        huber_loss = huber_loss + self.config['kappa'] * (u.abs() - u.abs().clamp(min=0., max=self.config['kappa']) - 0.5 * self.config['kappa'])
        quantile_loss = (tau.unsqueeze(0) - (u < 0).float()).abs() * huber_loss
        loss = quantile_loss.mean()
        return loss

    def learn(self):
        batch_size = self.config['batch_size']
        (observations, actions, rewards, next_observations, dones), weights = \
                                            self.memory.sample(batch_size)

        observations = torch.FloatTensor(observations)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).expand(batch_size, self.config['N_dash'])
        next_observations = torch.FloatTensor(next_observations)
        dones = torch.FloatTensor(dones).unsqueeze(1).expand(batch_size, self.config['N_dash'])
        tau_n = torch.rand((self.config['N']))
        tau_nDash = torch.rand((self.config['N_dash']))
        if self.USE_GPU:
            observations, actions, rewards, next_observations, dones, tau_n, tuan_nDash = observations.cuda(), \
                                                                                          actions.cuda(), rewards.cuda(), \
                                                                                          next_observations.cuda(), \
                                                                                          dones.cuda(), tau_n.cuda(), tau_nDash.cuda()
        # 获得该状态下的value和τ

        # Calculate quantile values of current states and actions at tau_hats.
        current_sa_quantiles = evaluate_quantile_at_action(
            self.pred_net(observations, tau_n)[0],
            actions).squeeze()

        with torch.no_grad():
            # Calculate Q values of next states.
            taus_ = torch.rand( batch_size, self.config['N'], dtype=observations.dtype,
                device=observations.device)
            next_q = self.pred_net(next_observations, taus_)[0].mean(dim=-2).squeeze()
            # Calculate greedy actions.
            next_actions = torch.argmax(next_q, dim=1, keepdim=True).squeeze().cuda()
            # assert next_actions.shape == (batch_size, 1)

            # Calculate quantile values of next states and next actions.
            next_sa_quantiles = evaluate_quantile_at_action(
                self.target_net(next_observations, tau_nDash)[0], next_actions).squeeze()
            # assert next_sa_quantiles.shape == (batch_size, 1, N_dash)

            # Calculate target quantile values.
            target_sa_quantiles = rewards + self.GAMMA * next_sa_quantiles * (1. - dones)
            # assert target_sa_quantiles.shape == (
            #     batch_size, 1, N_dash)

        td_errors = target_sa_quantiles.unsqueeze(1) - current_sa_quantiles.unsqueeze(-1)
        # assert td_errors.shape == (batch_size, N, N_dash)

        quantile_huber_loss = calculate_quantile_huber_loss(
            td_errors, tau_n, None, self.config['kappa'])
        update_params(
            self.optimizer, quantile_huber_loss,
            networks=[self.target_net],
            retain_graph=False, grad_cliping=5.0)

        if self.config['use_per']:
            self.memory.update_priority(td_errors)
        return quantile_huber_loss.item()

    def learn_new(self):
        if self.use_per:
            (states, actions, rewards, next_states, dones), weights =\
                self.memory.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones =\
                self.memory.sample(self.batch_size)
            weights = None

        quantile_loss, mean_q, errors = self.calculate_loss(
            states, actions, rewards, next_states, dones, weights)
        assert errors.shape == (self.batch_size, 1)
        update_params(
            self.optimizer, quantile_loss,
            networks=[self.pred_net],
            retain_graph=False, grad_cliping=5.0)

        if self.use_per:
            self.memory.update_priority(errors)
        return quantile_loss.detach().item()

    def calculate_loss(self, state, actions, rewards, next_states,
                       dones, weights):
        # Sample fractions.
        taus = torch.rand(
            self.batch_size, self.N, dtype=state.dtype,
            device=state.device)

        # Calculate quantile values of current states and actions at tau_hats.
        current_sa_quantiles = evaluate_quantile_at_action(
            self.pred_net(state, taus)[0], actions)
        assert current_sa_quantiles.shape == (
            self.batch_size, self.N, 1)

        with torch.no_grad():
            # Calculate Q values of next states.  batch N a
            next_q = self.target_net(next_states)[0].mean(dim=1)

            # Calculate greedy actions.
            next_actions = torch.argmax(next_q, dim=1, keepdim=True)
            assert next_actions.shape == (self.batch_size, 1)

            # Sample next fractions.
            tau_dashes = torch.rand(
                self.batch_size, self.N_dash, dtype=state.dtype,
                device=state.device)

            # Calculate quantile values of next states and next actions.
            next_sa_quantiles = evaluate_quantile_at_action(
                self.target_net(next_states,
                    tau_dashes)[0], next_actions).transpose(1, 2)
            assert next_sa_quantiles.shape == (self.batch_size, 1, self.N_dash)

            # Calculate target quantile values.
            target_sa_quantiles = rewards[..., None] + (
                1.0 - dones[..., None]) * self.GAMMA * next_sa_quantiles
            assert target_sa_quantiles.shape == (
                self.batch_size, 1, self.N_dash)

        td_errors = target_sa_quantiles - current_sa_quantiles
        assert td_errors.shape == (self.batch_size, self.N, self.N_dash)

        quantile_huber_loss = calculate_quantile_huber_loss(
            td_errors, taus, weights, self.kappa)

        return quantile_huber_loss, next_q.detach().mean().item(), \
            td_errors.detach().abs().sum(dim=1).mean(dim=1, keepdim=True)

    # Update target network
    def update_target(self):
        target = self.target_net
        pred = self.pred_net
        update_rate = 0.9
        # update target network parameters using predcition network
        for target_param, pred_param in zip(target.parameters(), pred.parameters()):
            target_param.data.copy_((1.0 - update_rate) \
                                    * target_param.data + update_rate * pred_param.data)

    def save_model(self):
        # save prediction network and target network
        self.pred_net.save(self.save_dir)
        self.target_net.save(self.save_dir)

    def load_model(self):
        # load prediction network and target network
        self.pred_net.load(self.save_dir)
        self.target_net.load(self.save_dir)

    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(
            self.pred_net.state_dict(),
            os.path.join(save_dir, 'online_net.pth'))
        torch.save(
            self.target_net.state_dict(),
            os.path.join(save_dir, 'target_net.pth'))

    def load_models(self, save_dir):
        self.pred_net.load_state_dict(torch.load(
            os.path.join(save_dir, 'online_net.pth')))
        self.target_net.load_state_dict(torch.load(
            os.path.join(save_dir, 'target_net.pth')))

