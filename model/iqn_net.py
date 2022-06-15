import random

import torch
from torch import nn
import torch.nn.functional as F

from model.cnn_net import CNN
from utils.config import config
import numpy as np

from utils.utils import get_optimistic


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class StateEmbedding(nn.Module):
    def __init__(self, num_channels, embedding_dim=7 * 7 * 64):
        super(StateEmbedding, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            Flatten(),
            # nn.Linear(embedding_dim, 512),
            # nn.ReLU(),
        ).apply(initialize_weights_he)

        self.embedding_dim = embedding_dim

        self.state_emb_net = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU()
        )

    def forward(self, states):
        batch_size = states.shape[0]

        # Calculate embeddings of states.
        state_embedding = self.net(states)
        assert state_embedding.shape == (batch_size, self.embedding_dim)

        return state_embedding

    def get_state_emb(self, states):
        batch_size = states.shape[0]

        # Calculate embeddings of states.
        state_embedding = self.net(states)
        state_embedding = self.state_emb_net(state_embedding)

        return state_embedding


class CosineEmbeddingNetwork(nn.Module):

    def __init__(self, num_inputs=None, num_cosines=64, embedding_dim=64):
        super(CosineEmbeddingNetwork, self).__init__()
        linear = nn.Linear

        self.net = nn.Sequential(
            linear(num_cosines, embedding_dim),
            nn.ReLU()
        )
        # self.feature = nn.Linear(num_inputs, num_cosines)
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim

    def forward(self, taus):
        if len(taus.shape) == 1:
            batch_size = 1
        else:
            batch_size = taus.shape[0]
        N = taus.shape[-1]
        # Calculate i * \pi (i=1,...,N).
        i_pi = np.pi * torch.arange(
            start=1, end=self.num_cosines+1, dtype=taus.dtype,
            device=taus.device).view(1, 1, self.num_cosines)

        # Calculate cos(i * \pi * \tau).
        cosines = torch.cos(
            taus.view(batch_size, N, 1) * i_pi
            ).view(batch_size * N, self.num_cosines).cuda()
        cosines = cosines.to(torch.float32)
        # Calculate embeddings of taus.
        tau_embeddings = self.net(cosines).view(
            batch_size, N, self.embedding_dim)
        # state = state.to(torch.float32)
        # state_emb = self.feature(state)

        return tau_embeddings


class Quantile_Net(nn.Module):
    def __init__(self, input_dim, num_actions, embedding_dim=64, dueling_net=False, noisy_net=False):
        super(Quantile_Net, self).__init__()
        linear = nn.Linear

        # if not dueling_net:
        self.net = nn.Sequential(
            linear(input_dim, embedding_dim),
            nn.ReLU(),
            linear(embedding_dim, num_actions),
        )
        # else:
        # 	self.advantage_net = nn.Sequential(
        # 		linear(embedding_dim, 512),
        # 		nn.ReLU(),
        # 		linear(512, num_actions),
        # 	)
        # 	self.baseline_net = nn.Sequential(
        # 		linear(embedding_dim, 512),
        # 		nn.ReLU(),
        # 		linear(512, 1),
        # 	)
        self.hidden_dim = embedding_dim
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.dueling_net = dueling_net
        self.noisy_net = noisy_net

    def forward(self, state_embeddings, tau_embeddings):
        if len(state_embeddings.shape)==1:
            batch_size = 1
        else:
            batch_size=state_embeddings.shape[0]

        if state_embeddings.shape[0] is not self.input_dim:
            state_embeddings = state_embeddings.view(batch_size, 1, self.input_dim)

        # if state_embeddings.shape[1] == tau_embeddings.shape[0]:
        #     state_embeddings = state_embeddings.transpose(0, 1).unsqueeze(2)
        # assert state_embeddings.shape[0] == tau_embeddings.shape[0]
        # assert state_embeddings.shape[2] == tau_embeddings.shape[2]

        # NOTE: Because variable taus correspond to either \tau or \hat \tau
        # in the paper, N isn't neccesarily the same as fqf.N.  1  50
        # batch_size = state_embeddings.shape[1]
        N = tau_embeddings.shape[1]
        # state_embeddings = state_embeddings.unsqueeze(-2)
        # Reshape into (batch_size, 1, embedding_dim). 1 1 5

        # state_embeddings = state_embeddings.viewew(n_agentent,
            # batch_size, 1, self.embedding_dim)

        # Calculate embeddings of states and taus.
        embeddings = (state_embeddings * tau_embeddings)

        # Calculate quantile values.
        # if not self.dueling_net:
        quantiles = self.net(embeddings)
        # else:
        # 	advantages = self.advantage_net(embeddings)
        # 	baselines = self.baseline_net(embeddings)
        # 	quantiles =\
        # 		baselines + advantages - advantages.mean(1, keepdim=True)
        # if batch_size != 1:
        # quantiles = quantiles.view(batch_size, self.num_actions, N)

        return quantiles


class Quantile_Net_init(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(Quantile_Net_init, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.feature_fc1 = nn.Linear(self.observation_dim, 64)
        # self.feature_fc2 = nn.Linear(64, 64)

        self.phi = nn.Linear(1, 64, bias=False)
        self.phi_bias = nn.Parameter(torch.zeros(64), requires_grad=True)

        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, self.action_dim)

    def forward(self, observation, tau):
        # RELU(4, 128)

        sample_num = observation.shape[0]
        x = F.relu(self.feature_fc1(observation))  # [1, 5]
        # relu(128,128)-----feature
        # x = F.relu(self.feature_fc2(x))

        # tau = torch.rand(sample_num, 1)  [50, 1]
        tau = tau
        # * tau is the quantile vector
        quants = torch.arange(0, tau.shape[0], 1.0)  # [50, 1] [0 ,1, 2, ...., 49]
        if config.use_gpu:
            tau = tau.cuda()
            quants = quants.cuda()
        cos_trans = torch.cos(np.pi * quants * tau).unsqueeze(2)
        # * cos_trans: [sample_num, sample_num, 1]
        rand_feat = F.relu(self.phi(cos_trans).mean(1) + self.phi_bias.unsqueeze(0)).unsqueeze(0)
        #  100,5,50,64 ----->5,50,    *  100,1,64
        # * rand_feat: [1, sample_num, 64]
        x = x.unsqueeze(1)
        # * x: [batch_size, 1, 64]
        x = x * rand_feat
        # * x: [batch_size, sample_num, 64]
        x = F.relu(self.fc1(x))
        # * x: [batch_size, sample_num, 64]
        value = self.fc2(x).transpose(1, 2)
        # * value: [batch_size, action_dim, sample_num]
        return value, tau

    def get_value(self, observation, tau=None):
        if tau is None:
            tau = torch.rand(config.N, 1)
        value, tau = self.forward(observation, tau)
        return value, tau

    def act(self, observation, tau):
        value, tau = self.forward(observation, tau)
        # * beta is set to be an identity mapping here so calculate the expectation
        action = value.mean(2).max(1)[1].detach().item()
        return action

    def save(self, PATH):
        torch.save(self.state_dict(), PATH)

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))


class iqn(nn.Module):
    def __init__(self, num_inputs, hidden_dim, num_actions, num_cosines=64):
        super(iqn, self).__init__()

        # self.encoder = Encoder(num_inputs, hidden_dim)
        # self.att_1 = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim)
        # self.att_2 = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim)
        # self.q_net = Q_Net(hidden_dim, num_actions)
        self.cnn_net = CNN(num_inputs)
        self.cosine_net = CosineEmbeddingNetwork(num_cosines=num_cosines, embedding_dim=hidden_dim)
        self.q_net = Quantile_Net(hidden_dim, num_actions)

    def forward(self, state, tau=None):
        # h1 = self.encoder(x)
        # h2 = self.att_1(h1, mask)
        # h3 = self.att_2(h2, mask)
        # q = self.q_net(h3)
        # q = torch.zeros((h2.shape[0], h2.shape[1], tau.shape[-1], action_dim))
        # batch_size
        state_emb = self.cnn_net(state)
        norm_opt = 0
        if tau is None:
            tau, norm_opt = get_optimistic(state_emb.cpu().detach().numpy()[0])
        tau_embeeding = self.cosine_net(torch.tensor(tau).cuda())
        q = self.q_net(state_emb, tau_embeeding)

        return q, norm_opt