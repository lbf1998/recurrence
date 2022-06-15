import os
import random
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
import time

from agent.iqn_sim_agent import IQN_SIM
from env.env import make_pytorch_env
from utils.util import LinearSchedule
from utils.utils import RunningMeanStats

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
np.seterr(divide='ignore', invalid='ignore')

with open('../agent/iqn.yaml') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
f.close()

summary_dir = '../log/summary/'
if not os.path.exists(summary_dir):
    os.makedirs(summary_dir)
writer = SummaryWriter(log_dir=summary_dir)
train_return = RunningMeanStats(config['log_interval'])

MAX_STEPS = config['num_steps']
MAX_EPISODE_STEPS = config['max_episode_steps']
TRAIN = True
# TRAIN = FalseEnv
RANDOM_EPISODE = config['random_starts']


def set_seed(lucky_number):
    torch.manual_seed(lucky_number)
    torch.cuda.manual_seed_all(lucky_number)
    np.random.seed(lucky_number)
    random.seed(lucky_number)


def test(test_env, ag1, steps, load_model=False):
    num_episodes = 0
    num_steps = 0
    total_return = 0.0

    while True:
        state = test_env.reset()
        episode_steps = 0
        episode_return = 0.0
        done = False
        while (not done) and episode_steps <= config['max_episode_steps']:
            state = torch.ByteTensor(
                state).unsqueeze(0).float() / 255.
            action, _ = ag1.test_action(state)

            next_state, reward, done, _ = test_env.step(action)
            num_steps += 1
            episode_steps += 1
            episode_return += reward
            state = next_state

        num_episodes += 1
        total_return += episode_return

        if num_steps > config['num_eval_steps']:
            break

    mean_return = total_return / num_episodes

    if mean_return > ag1.best_eval_score:
        ag1.best_eval_score = mean_return
        ag1.save_models(os.path.join(ag1.save_dir, 'best'))

    # We log evaluation results along with training frames = 4 * steps.
    writer.add_scalar(
        'return/test', mean_return, 4 * steps)
    print('-' * 60)
    print(f'Num steps: {steps:<5}  '
          f'return: {mean_return:<5.1f}')
    print('-' * 60)


def main():
    # env = ShipAcrossRiverHarderD(MontezumaRevenge)
    # test_env = ShipAcrossRiverHarderD()
    ENV_NAME = 'MsPacman-v0'
    env = make_pytorch_env(ENV_NAME)
    test_env = make_pytorch_env(ENV_NAME, episode_life=False, clip_rewards=False)
    lucky_no = 4
    set_seed(lucky_no)
    loss1 = 0
    # erm_size = 100000
    # print(type(env.observation_space.shape))

    agent1 = IQN_SIM(env, logdir='iqn_logs1', savedir='iqn_save1')
    # agent2 = IQN_SIM(state_dim, action_dim, logdir='iqnologs2', savedir='iqnosave2')

    print('after init')
    train_log = []
    test_log = []
    tau = []
    train_episode = 0
    steps = 0
    while True:
        train_episode += 1
        state = env.reset()
        epsilon_schedule = LinearSchedule(schedule_timesteps=config['epsilon_decay_steps'],
                                          final_p=config['epsilon_train'],
                                          initial_p=1.0)
        episode_len = 0
        episode_reward = 0
        # episode1, episode2 = [], []

        for j in range(MAX_EPISODE_STEPS):
            if np.random.rand() < epsilon_schedule.value(steps):
                action = env.action_space.sample()
            else:
                state = torch.ByteTensor(
                    state).unsqueeze(0).float() / 255.
                action, tau_1 = agent1.op_choose_action(state)

            # epilon = epsilon_schedule.value(i)
            # get observations to input to Q network (need to append prev frames)
            # action1 = agent1.test_action(state)
            # action2, _ = agent2.op_choose_action(state)
            next_state, reward, done, info = env.step(action)

            # reward = np.clip(reward, -1.0, 1.0)

            agent1.memory.append(state, action, reward, next_state, done)
            steps += 1
            episode_len += 1
            episode_reward += reward
            # state1 = next_state
            state = next_state

            if steps > config['start_steps'] and steps % config['update_interval'] == 0:
                loss1 = agent1.learn_new()

            if steps % config['target_update_interval'] == 0:
                agent1.update_target()

            if steps % config['eval_interval'] == 0:
                print('testing...')
                test(test_env, agent1, steps, load_model=False)
                agent1.save_models(os.path.join(agent1.save_dir, 'final'))

            if done:
                # agent2.learn()
                # tau.append(tau_1)
                this_train_log = (steps, train_episode, episode_len, loss1, episode_reward)
                train_log.append(this_train_log)
                train_return.append(episode_reward)
                print(
                    'steps:{}, episode: {}, len: {},loss1: {}, reward: {}'.format(
                        *this_train_log))
                break

        if train_episode % config['log_interval'] == 0:
            writer.add_scalar(
                'return/train', train_return.get(), 4 * steps)

        if steps > MAX_STEPS:
            break

    env.close()
    test_env.close()
    writer.close()


if __name__ == '__main__':
    main()
