from collections import deque

import numpy as np
import pandas as pd
import math
import torch
from utils.config import config
# from config import *

# 初始化正交矩阵
a = np.zeros((config.k,config.D))
for i in range(config.k):
    for j in range(config.D):
        a[i][j] = np.random.normal()

hash_arr = {}


def simhash(state):
    print(state)
    state = state.transpose(-1, -2)
    fhi = np.sign(np.matmul(a, state))
    for j in range(len(fhi[0])):
        sign = ""
        for k in range(len(fhi)):
            sign += fhi[k][j]
        if hash_arr[sign]:
            hash_arr[sign] = hash_arr[sign] + 1
        else:
            hash_arr[sign] = 1


def get_optimistic(state, K_sample=config.K):
    # state = state.squeeze().transpose(-1, -2).cpu().detach().numpy()
    fhi = np.sign(np.matmul(a, state))
    # n_fhi = np.zeros((1, 1))
    # for j in range(len(fhi[0])):
    sign = ""
    for k in range(len(fhi)):
        sign += str(int(fhi[k]))
    if hash_arr.__contains__(sign):
        hash_arr[sign] = hash_arr[sign] + 1
    else:
        hash_arr[sign] = 1
    # n_fhi 状态离散化之后的标志位
    n_fhi = math.atan(config.beta/math.sqrt(hash_arr.get(sign, 1)))

    norm_opt = n_fhi
    # taus = np.zeros((1, config.K))

    # for m in range(len(norm_opt)):
            #  利用乐观值进行采样tau
    taus = norm_opt + (1 - norm_opt) * np.random.random(K_sample)

    return taus, norm_opt


def preprocess_state(s):
    """
    preprocess gym images before storing them or passing them through the network.
    - from rgb to grayscale
    - normalize
    - crop
    - permute (h, w, c) to (c, h, w) as pytorch expects
    - to tensor
    """

    # transofrm image to grayscale
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3],
                      [0.2989, 0.5870, 0.1140])[..., np.newaxis] / 255

    state = rgb2gray(s.copy())
    # create tensor crop and permute image
    states = torch.from_numpy(state[15:200, 30:125, :].transpose(
        2, 0, 1)).float().unsqueeze(0)

    return states


def disable_gradients(network):
    # Disable calculations of gradients.
    for param in network.parameters():
        param.requires_grad = False


def calculate_huber_loss(td_errors, kappa=1.0):
    return torch.where(
        td_errors.abs() <= kappa,
        0.5 * td_errors.pow(2),
        kappa * (td_errors.abs() - 0.5 * kappa))


def calculate_quantile_huber_loss(td_errors, taus, weights=None, kappa=1.0):
    assert not taus.requires_grad
    batch_size, N, N_dash = td_errors.shape

    # Calculate huber loss element-wisely.
    element_wise_huber_loss = calculate_huber_loss(td_errors)
    # assert element_wise_huber_loss.shape == (
    #     batch_size, N, N_dash)

    # Calculate quantile huber loss element-wisely.
    element_wise_quantile_huber_loss = torch.abs(
        taus[:, ..., None] - (td_errors.detach() < 0).float()
        ) * element_wise_huber_loss / kappa
    # assert element_wise_quantile_huber_loss.shape == (
    #     batch_size, N, N_dash)

    # Quantile huber loss.
    batch_quantile_huber_loss = element_wise_quantile_huber_loss.sum(
        dim=2).mean(dim=1, keepdim=True)
    # assert batch_quantile_huber_loss.shape == (n_agent, batch_size, 1)

    if weights is not None:
        quantile_huber_loss = (batch_quantile_huber_loss * weights).mean()
    else:
        quantile_huber_loss = batch_quantile_huber_loss.mean()

    return quantile_huber_loss


def evaluate_quantile_at_action(s_quantiles, actions):
    assert s_quantiles.shape[0] == actions.shape[0]

    batch_size = s_quantiles.shape[0]
    N = s_quantiles.shape[1]

    # Expand actions into (batch_size, N, 1).
    action_index = actions[:, ..., None, None]
    # Expand actions into (batch_size, N, 1).
    action_index = actions[..., None].expand(batch_size, N, 1)

    # Calculate quantile values at specified actions.
    sa_quantiles = s_quantiles.gather(dim=2, index=action_index)
    # if len(action_index.shape) == len(s_quantiles.shape):
    #     action_index = action_index.expand(batch_size, N, 1).to(torch.int64)
    # # Calculate quantile values at specified actions.
    # else:
    #     action_index = action_index.squeeze(-1).expand(100, batch_size, N, 1).to(torch.int64)
    #
    # sa_quantiles = s_quantiles.gather(dim=2, index=action_index)

    return sa_quantiles


def update_params(optim, loss, networks, retain_graph=False,
                  grad_cliping=None):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    # Clip norms of gradients to stebilize training.
    if grad_cliping:
        for net in networks:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_cliping)
    optim.step()


class RunningMeanStats:

    def __init__(self, n=10):
        self.n = n
        self.stats = deque(maxlen=n)

    def append(self, x):
        self.stats.append(x)

    def get(self):
        return np.mean(self.stats)


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.

        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)



