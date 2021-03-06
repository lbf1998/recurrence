#coding=utf-8

import gym
from enum import Enum
import numpy as np
import random

class CMOTP_IL(gym.Env):
    """
    if used random policy, the expect of episode length is about 1400,
    the expect of episode length of task 1(grasp the good) is about 887,
    eht expect of episode length of task 2(transport the good to the home) is about 515

    action: 0 -> stay, 1 -> north, 2 -> east, 3 -> south, 4 -> west
    """

    GRASP_STATE = Enum('GRASP_STATE', ('FREE', 'GRASPING_LEFT', 'GRASPING_RIGHT'))

    def __init__(self):
        self.move_delta = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
        self.viewer = None
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 1, 0, 0, 1]), high=np.array([11, 11, 3, 11, 11, 3]),
                                                dtype=np.int32)

    def reset(self):
        """
        average episode length with range strategy is *****.
        :return:
        """
        # 1 -> Agent1, 2 -> Agent2, 0 -> passibal region, -1 -> walls, 3 -> goods
        self.map = [[-1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [-1, -1, -1, -1, 0, 0, 0, -1, -1, -1, -1, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [-1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -1]]
        self.map_size = (len(self.map), len(self.map[0]))
        self.home_region = [(0, 3), (0, 4), (0, 5), (0, 6), (0, 7)]
        return np.float32(np.array((11, 0, self.GRASP_STATE.FREE.value, 11, 10, self.GRASP_STATE.FREE.value))), \
               np.float32(np.array((11, 0, self.GRASP_STATE.FREE.value, 11, 10, self.GRASP_STATE.FREE.value)))
    '''
    def __init4__(self):
        self.move_delta = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
        self.viewer = None
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 1, 0, 0, 1]), high=np.array([7, 12, 3, 7, 12, 3]),
                                                dtype=np.int32)

    def reset4(self):
        # 1 -> Agent1, 2 -> Agent2, 0 -> passibal region, -1 -> walls, 3 -> goods
        self.map = [[-1, -1, 0, 0, 0, -1, -1, -1, 0, 0, 0, -1, -1],
                    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [-1, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 0, -1],
                    [-1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, -1],
                    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1],
                    [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

        self.map_size = (len(self.map), len(self.map[0]))
        self.home_region_1 = [(0, 2), (0, 3), (0, 4)]
        self.home_region_2 = [(0, 8), (0, 9), (0, 10)]
        return np.float32(np.array((6, 1, self.GRASP_STATE.FREE.value, 6, 11, self.GRASP_STATE.FREE.value))), \
               np.float32(np.array((6, 1, self.GRASP_STATE.FREE.value, 6, 11, self.GRASP_STATE.FREE.value)))
    '''

    '''
        def __init1__(self):
        self.move_delta = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
        self.viewer = None
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 1, 0, 0, 1]), high=np.array([11, 11, 3, 11, 11, 3]),
                                                dtype=np.int32)

    def reset1(self):
        """
        average episode length with range strategy is *****.
        :return:
        """
        # 1 -> Agent1, 2 -> Agent2, 0 -> passibal region, -1 -> walls, 3 -> goods
        self.map = [[-1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [-1, -1, -1, -1, -1, 0, 0, 0, -1, -1, -1, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [-1, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [-1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -1]]
        self.map_size = (len(self.map), len(self.map[0]))
        self.home_region = [(0, 3), (0, 4), (0, 5), (0, 6), (0, 7)]
        return np.float32(np.array((11, 0, self.GRASP_STATE.FREE.value, 11, 10, self.GRASP_STATE.FREE.value))), \
               np.float32(np.array((11, 0, self.GRASP_STATE.FREE.value, 11, 10, self.GRASP_STATE.FREE.value)))

    
    '''

    def __init1__(self):
        self.move_delta = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
        self.viewer = None
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 1, 0, 0, 1]), high=np.array([15, 15, 3, 15, 15, 3]),
                                                dtype=np.int32)

    def reset1(self):
        """
        average episode length with range strategy is *****.
        :return:
        """
        # 1 -> Agent1, 2 -> Agent2, 0 -> passibal region, -1 -> walls, 3 -> goods
        self.map = [[-1, -1, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [-1, -1, -1, -1, -1, -1, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [-1, -1, -1, -1, -1, -1, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -1]]
        self.map_size = (len(self.map), len(self.map[0]))
        self.home_region = [(0, 5), (0, 6), (0, 7), (0, 8), (0, 9)]
        return np.float32(np.array((15, 0, self.GRASP_STATE.FREE.value, 15, 14, self.GRASP_STATE.FREE.value))), \
               np.float32(np.array((15, 0, self.GRASP_STATE.FREE.value, 15, 14, self.GRASP_STATE.FREE.value)))

    def __init2__(self):
        self.move_delta = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
        self.viewer = None
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 1, 0, 0, 1]), high=np.array([15, 15, 3, 15, 15, 3]),
                                                dtype=np.int32)

    def reset2(self):
        # 1 -> Agent1, 2 -> Agent2, 0 -> passibal region, -1 -> walls, 3 -> goods
        self.map = [[-1, -1, -1, -1, 0, 0, 0, -1, 0, 0, 0, -1, -1, -1, -1, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [-1, -1, -1, -1, -1, -1, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -1]]
        self.map_size = (len(self.map), len(self.map[0]))
        self.home_region_1 = [(0, 4), (0, 5), (0, 6)]
        self.home_region_2 = [(0, 8), (0, 9), (0, 10)]
        return np.float32(np.array((15, 0, self.GRASP_STATE.FREE.value, 15, 14, self.GRASP_STATE.FREE.value))), \
               np.float32(np.array((15, 0, self.GRASP_STATE.FREE.value, 15, 14, self.GRASP_STATE.FREE.value)))

    def __init4__(self):
        self.move_delta = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
        self.viewer = None
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 1, 0, 0, 1]), high=np.array([7, 12, 3, 7, 12, 3]),
                                                dtype=np.int32)

    def reset4(self):
        # 1 -> Agent1, 2 -> Agent2, 0 -> passibal region, -1 -> walls, 3 -> goods
        self.map = [[-1, -1, 0, 0, 0, -1, -1, -1, 0, 0, 0, -1, -1],
                    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [-1, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 0, -1],
                    [-1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, -1],
                    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1],
                    [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

        self.map_size = (len(self.map), len(self.map[0]))
        self.home_region_1 = [(0, 2), (0, 3), (0, 4)]
        self.home_region_2 = [(0, 8), (0, 9), (0, 10)]
        return np.float32(np.array((6, 1, self.GRASP_STATE.FREE.value, 6, 11, self.GRASP_STATE.FREE.value))), \
               np.float32(np.array((6, 1, self.GRASP_STATE.FREE.value, 6, 11, self.GRASP_STATE.FREE.value)))

    def __init5__(self):
        self.move_delta = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
        self.viewer = None
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 1, 0, 0, 1]), high=np.array([13, 12, 3, 13, 12, 3]),
                                                dtype=np.int32)

    def reset5(self):
        # 1 -> Agent1, 2 -> Agent2, 0 -> passibal region, -1 -> walls, 3 -> goods
        self.map = [[-1, -1, 0, 0, 0, -1, -1, -1, 0, 0, 0, -1, -1],
                    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [-1, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 0, -1],
                    [-1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, -1],
                    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                    [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1],
                    [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

        self.map_size = (len(self.map), len(self.map[0]))
        self.home_region_1 = [(0, 2), (0, 3), (0, 4)]
        self.home_region_2 = [(0, 8), (0, 9), (0, 10)]
        return np.float32(np.array((12, 1, self.GRASP_STATE.FREE.value, 12, 11, self.GRASP_STATE.FREE.value))), \
               np.float32(np.array((12, 1, self.GRASP_STATE.FREE.value, 12, 11, self.GRASP_STATE.FREE.value)))


    def step_1(self, action_n):
        # validate actions
        for i, ac in enumerate(action_n):
            assert ac >= 0 and ac <= 4, "agent {}'s action is out of range.".format(i + 1)

        pos_agent1, pos_agent2, pos_goods = self.get_entity_pos()
        grasp_before_move = self.grasp_goods(pos_agent1, pos_agent2, pos_goods)
        self.move(action_n)
        new_pos_agent1, new_pos_agent2, new_pos_goods = self.get_entity_pos()
        grasp_after_move = self.grasp_goods(new_pos_agent1, new_pos_agent2, new_pos_goods)

        agent1_grasp_state = self.GRASP_STATE.FREE
        agent2_grasp_state = self.GRASP_STATE.FREE

        if grasp_after_move:
            # ???????????????????????????????????????
            if new_pos_agent1[1] + 1 == new_pos_goods[1]:
                agent1_grasp_state = self.GRASP_STATE.GRASPING_LEFT
                agent2_grasp_state = self.GRASP_STATE.GRASPING_RIGHT
            elif new_pos_agent1[1] - 1 == new_pos_goods[1]:
                agent1_grasp_state = self.GRASP_STATE.GRASPING_RIGHT
                agent2_grasp_state = self.GRASP_STATE.GRASPING_LEFT

        terminate = False

        # calculate reward
        reward = 0.

        if not grasp_before_move and grasp_after_move:
            reward += 1.
        if self.home_region.__contains__(new_pos_goods):
            reward += 8.
            terminate = True

        # return (self.map, self.map), (reward, reward), (terminate, terminate), ({}, {})
        return (np.float32(np.array((*new_pos_agent1, agent1_grasp_state.value, *new_pos_agent2, agent2_grasp_state.value))),
                np.float32(np.array((*new_pos_agent1, agent1_grasp_state.value, *new_pos_agent2, agent2_grasp_state.value)))), \
               (reward, reward), (terminate, terminate), ({}, {})

    def step_2(self, action_n):
        # validate actions
        for i, ac in enumerate(action_n):
            assert ac >= 0 and ac <= 4, "agent {}'s action is out of range.".format(i + 1)

        pos_agent1, pos_agent2, pos_goods = self.get_entity_pos()
        grasp_before_move = self.grasp_goods(pos_agent1, pos_agent2, pos_goods)
        self.move(action_n)
        new_pos_agent1, new_pos_agent2, new_pos_goods = self.get_entity_pos()
        grasp_after_move = self.grasp_goods(new_pos_agent1, new_pos_agent2, new_pos_goods)

        agent1_grasp_state = self.GRASP_STATE.FREE
        agent2_grasp_state = self.GRASP_STATE.FREE

        if grasp_after_move:
            # ???????????????????????????????????????
            if new_pos_agent1[1] + 1 == new_pos_goods[1]:
                agent1_grasp_state = self.GRASP_STATE.GRASPING_LEFT
                agent2_grasp_state = self.GRASP_STATE.GRASPING_RIGHT
            elif new_pos_agent1[1] - 1 == new_pos_goods[1]:
                agent1_grasp_state = self.GRASP_STATE.GRASPING_RIGHT
                agent2_grasp_state = self.GRASP_STATE.GRASPING_LEFT

        terminate = False

        # calculate reward
        reward = 0.
        info = 0
        if not grasp_before_move and grasp_after_move:
            reward += 1.
        if self.home_region_1.__contains__(new_pos_goods):
            reward += 8.
            terminate = True
            info = 1
        if self.home_region_2.__contains__(new_pos_goods):
            reward += 8.
            terminate = True
            info = 2

        # return (self.map, self.map), (reward, reward), (terminate, terminate), ({}, {})
        return (np.float32(np.array((*new_pos_agent1, agent1_grasp_state.value, *new_pos_agent2, agent2_grasp_state.value))),
                np.float32(np.array((*new_pos_agent1, agent1_grasp_state.value, *new_pos_agent2, agent2_grasp_state.value)))), \
               (reward, reward), (terminate, terminate), info

    def step_3(self, action_n):
        # validate actions
        for i, ac in enumerate(action_n):
            assert ac >= 0 and ac <= 4, "agent {}'s action is out of range.".format(i + 1)

        pos_agent1, pos_agent2, pos_goods = self.get_entity_pos()
        grasp_before_move = self.grasp_goods(pos_agent1, pos_agent2, pos_goods)
        self.move(action_n)
        new_pos_agent1, new_pos_agent2, new_pos_goods = self.get_entity_pos()
        grasp_after_move = self.grasp_goods(new_pos_agent1, new_pos_agent2, new_pos_goods)

        agent1_grasp_state = self.GRASP_STATE.FREE
        agent2_grasp_state = self.GRASP_STATE.FREE

        if grasp_after_move:
            # ???????????????????????????????????????
            if new_pos_agent1[1] + 1 == new_pos_goods[1]:
                agent1_grasp_state = self.GRASP_STATE.GRASPING_LEFT
                agent2_grasp_state = self.GRASP_STATE.GRASPING_RIGHT
            elif new_pos_agent1[1] - 1 == new_pos_goods[1]:
                agent1_grasp_state = self.GRASP_STATE.GRASPING_RIGHT
                agent2_grasp_state = self.GRASP_STATE.GRASPING_LEFT

        terminate = False

        # calculate reward
        reward = 0.
        info = 0
        if not grasp_before_move and grasp_after_move:
            reward += 1.
        if self.home_region_1.__contains__(new_pos_goods):
            reward += 8.
            terminate = True
            info = 1
        if self.home_region_2.__contains__(new_pos_goods):
            if random.random() > 0.5:
                reward += 12
            else:
                reward += 0
            terminate = True
            info = 2

        # return (self.map, self.map), (reward, reward), (terminate, terminate), ({}, {})
        return (np.float32(np.array((*new_pos_agent1, agent1_grasp_state.value, *new_pos_agent2, agent2_grasp_state.value))),
                np.float32(np.array((*new_pos_agent1, agent1_grasp_state.value, *new_pos_agent2, agent2_grasp_state.value)))), \
               (reward, reward), (terminate, terminate), info

    def render1(self, mode='human'):
        from gym.envs.classic_control import rendering
        grid_len = 50.
        screen_width = grid_len * self.map_size[1]
        screen_height = grid_len * self.map_size[0]

        def map_coor(pos):
            """
            self.map??????????????????self.viewer????????????
            :param pos: ??????????????????????????????tuple
            :return:
            """
            return (pos[1] + 0.5) * grid_len, (self.map_size[0] - 1 - pos[0] + 0.5) * grid_len

        def make_rectangle(pos):
            """
            ??????????????????????????????????????????????????????4???????????????
            :param pos: ??????????????????????????????tuple
            :return:
            """
            center_x, center_y = map_coor(pos)
            return [(center_x - grid_len / 2, center_y - grid_len / 2),
                    (center_x - grid_len / 2, center_y + grid_len / 2),
                    (center_x + grid_len / 2, center_y + grid_len / 2),
                    (center_x + grid_len / 2, center_y - grid_len / 2)]

        if self.viewer is None:
            self.viewer = rendering.Viewer(int(screen_width), int(screen_height))

            # draw lines
            lines_view = []
            lines_view.append(rendering.Line((0., 0.), (screen_width, 0.)))
            lines_view.append(rendering.Line((0., 0.), (0., screen_height)))

            for i in range(self.map_size[0]):
                lines_view.append(rendering.Line((0., (i + 1) * grid_len), (screen_width, (i + 1) * grid_len)))

            for i in range(self.map_size[1]):
                lines_view.append(rendering.Line(((i + 1) * grid_len, 0.), ((i + 1) * grid_len, screen_height)))

            for i in range(len(lines_view)):
                lines_view[i].set_color(0, 0, 0)

            # draw obstacles
            obstacles_view = []
            for i in range(self.map_size[0]):
                for j in range(self.map_size[1]):
                    if self.map[i][j] == -1:
                        obstcl = rendering.make_polygon(make_rectangle((i, j)))
                        obstcl.set_color(0, 0, 0)
                        obstacles_view.append(obstcl)
            home_regions_view = []

            for home in self.home_region:
                home_view = rendering.make_polygon(make_rectangle((home[0], home[1])))
                home_view.set_color(100, 100, 0)
                home_regions_view.append(home_view)


            # draw agents and goods

            agent1_view = rendering.make_circle(20)
            self.agent1_trans = rendering.Transform()
            agent1_view.add_attr(self.agent1_trans)
            agent1_view.set_color(1, 0, 0)

            agent2_view = rendering.make_circle(20)
            self.agent2_trans = rendering.Transform()
            agent2_view.add_attr(self.agent2_trans)
            agent2_view.set_color(0, 1, 0)

            goods_view = rendering.make_circle(15)
            self.goods_trans = rendering.Transform()
            goods_view.add_attr(self.goods_trans)
            goods_view.set_color(0, 0, 1)

            for i in range(len(lines_view)):
                self.viewer.add_geom(lines_view[i])
            self.viewer.add_geom(agent1_view)
            self.viewer.add_geom(agent2_view)
            self.viewer.add_geom(goods_view)
            for i in obstacles_view:
                self.viewer.add_geom(i)
            for i in home_regions_view:
                self.viewer.add_geom(i)

        pos_agent1, pos_agent2, pos_goods = self.get_entity_pos()

        self.agent1_trans.set_translation(*map_coor(pos_agent1))
        self.agent2_trans.set_translation(*map_coor(pos_agent2))
        self.goods_trans.set_translation(*map_coor(pos_goods))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def render2(self, mode='human'):
        from gym.envs.classic_control import rendering
        grid_len = 50.
        screen_width = grid_len * self.map_size[1]
        screen_height = grid_len * self.map_size[0]

        def map_coor(pos):
            """
            self.map??????????????????self.viewer????????????
            :param pos: ??????????????????????????????tuple
            :return:
            """
            return (pos[1] + 0.5) * grid_len, (self.map_size[0] - 1 - pos[0] + 0.5) * grid_len

        def make_rectangle(pos):
            """
            ??????????????????????????????????????????????????????4???????????????
            :param pos: ??????????????????????????????tuple
            :return:
            """
            center_x, center_y = map_coor(pos)
            return [(center_x - grid_len / 2, center_y - grid_len / 2),
                    (center_x - grid_len / 2, center_y + grid_len / 2),
                    (center_x + grid_len / 2, center_y + grid_len / 2),
                    (center_x + grid_len / 2, center_y - grid_len / 2)]

        if self.viewer is None:
            self.viewer = rendering.Viewer(int(screen_width), int(screen_height))

            # draw lines
            lines_view = []
            lines_view.append(rendering.Line((0., 0.), (screen_width, 0.)))
            lines_view.append(rendering.Line((0., 0.), (0., screen_height)))

            for i in range(self.map_size[0]):
                lines_view.append(rendering.Line((0., (i + 1) * grid_len), (screen_width, (i + 1) * grid_len)))

            for i in range(self.map_size[1]):
                lines_view.append(rendering.Line(((i + 1) * grid_len, 0.), ((i + 1) * grid_len, screen_height)))

            for i in range(len(lines_view)):
                lines_view[i].set_color(0, 0, 0)

            # draw obstacles
            obstacles_view = []
            for i in range(self.map_size[0]):
                for j in range(self.map_size[1]):
                    if self.map[i][j] == -1:
                        obstcl = rendering.make_polygon(make_rectangle((i, j)))
                        obstcl.set_color(0, 0, 0)
                        obstacles_view.append(obstcl)
            home_regions_view = []

            #for home in self.home_region:
            #    home_view = rendering.make_polygon(make_rectangle((home[0], home[1])))
            #    home_view.set_color(100, 100, 0)
            #    home_regions_view.append(home_view)

            for home in self.home_region_1:
                home_view = rendering.make_polygon(make_rectangle((home[0], home[1])))
                home_view.set_color(100, 100, 0)
                home_regions_view.append(home_view)

            for home in self.home_region_2:
                home_view = rendering.make_polygon(make_rectangle((home[0], home[1])))
                home_view.set_color(100, 100, 0)
                home_regions_view.append(home_view)
            # draw agents and goods

            agent1_view = rendering.make_circle(20)
            self.agent1_trans = rendering.Transform()
            agent1_view.add_attr(self.agent1_trans)
            agent1_view.set_color(1, 0, 0)

            agent2_view = rendering.make_circle(20)
            self.agent2_trans = rendering.Transform()
            agent2_view.add_attr(self.agent2_trans)
            agent2_view.set_color(0, 1, 0)

            goods_view = rendering.make_circle(15)
            self.goods_trans = rendering.Transform()
            goods_view.add_attr(self.goods_trans)
            goods_view.set_color(0, 0, 1)

            for i in range(len(lines_view)):
                self.viewer.add_geom(lines_view[i])
            self.viewer.add_geom(agent1_view)
            self.viewer.add_geom(agent2_view)
            self.viewer.add_geom(goods_view)
            for i in obstacles_view:
                self.viewer.add_geom(i)
            for i in home_regions_view:
                self.viewer.add_geom(i)

        pos_agent1, pos_agent2, pos_goods = self.get_entity_pos()

        self.agent1_trans.set_translation(*map_coor(pos_agent1))
        self.agent2_trans.set_translation(*map_coor(pos_agent2))
        self.goods_trans.set_translation(*map_coor(pos_goods))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def render3(self, mode='human'):
        from gym.envs.classic_control import rendering
        grid_len = 50.
        screen_width = grid_len * self.map_size[1]
        screen_height = grid_len * self.map_size[0]

        def map_coor(pos):
            """
            self.map??????????????????self.viewer????????????
            :param pos: ??????????????????????????????tuple
            :return:
            """
            return (pos[1] + 0.5) * grid_len, (self.map_size[0] - 1 - pos[0] + 0.5) * grid_len

        def make_rectangle(pos):
            """
            ??????????????????????????????????????????????????????4???????????????
            :param pos: ??????????????????????????????tuple
            :return:
            """
            center_x, center_y = map_coor(pos)
            return [(center_x - grid_len / 2, center_y - grid_len / 2),
                    (center_x - grid_len / 2, center_y + grid_len / 2),
                    (center_x + grid_len / 2, center_y + grid_len / 2),
                    (center_x + grid_len / 2, center_y - grid_len / 2)]

        if self.viewer is None:
            self.viewer = rendering.Viewer(int(screen_width), int(screen_height))

            # draw lines
            lines_view = []
            lines_view.append(rendering.Line((0., 0.), (screen_width, 0.)))
            lines_view.append(rendering.Line((0., 0.), (0., screen_height)))

            for i in range(self.map_size[0]):
                lines_view.append(rendering.Line((0., (i + 1) * grid_len), (screen_width, (i + 1) * grid_len)))

            for i in range(self.map_size[1]):
                lines_view.append(rendering.Line(((i + 1) * grid_len, 0.), ((i + 1) * grid_len, screen_height)))

            for i in range(len(lines_view)):
                lines_view[i].set_color(0, 0, 0)

            # draw obstacles
            obstacles_view = []
            for i in range(self.map_size[0]):
                for j in range(self.map_size[1]):
                    if self.map[i][j] == -1:
                        obstcl = rendering.make_polygon(make_rectangle((i, j)))
                        obstcl.set_color(0, 0, 0)
                        obstacles_view.append(obstcl)
            home_regions_view = []

            #for home in self.home_region:
            #    home_view = rendering.make_polygon(make_rectangle((home[0], home[1])))
            #    home_view.set_color(100, 100, 0)
            #    home_regions_view.append(home_view)

            for home in self.home_region_1:
                home_view = rendering.make_polygon(make_rectangle((home[0], home[1])))
                home_view.set_color(100, 100, 50)
                home_regions_view.append(home_view)

            for home in self.home_region_2:
                home_view = rendering.make_polygon(make_rectangle((home[0], home[1])))
                home_view.set_color(100, 100, 0)
                home_regions_view.append(home_view)
            # draw agents and goods

            agent1_view = rendering.make_circle(20)
            self.agent1_trans = rendering.Transform()
            agent1_view.add_attr(self.agent1_trans)
            agent1_view.set_color(1, 0, 0)

            agent2_view = rendering.make_circle(20)
            self.agent2_trans = rendering.Transform()
            agent2_view.add_attr(self.agent2_trans)
            agent2_view.set_color(0, 1, 0)

            goods_view = rendering.make_circle(15)
            self.goods_trans = rendering.Transform()
            goods_view.add_attr(self.goods_trans)
            goods_view.set_color(0, 0, 1)

            for i in range(len(lines_view)):
                self.viewer.add_geom(lines_view[i])
            self.viewer.add_geom(agent1_view)
            self.viewer.add_geom(agent2_view)
            self.viewer.add_geom(goods_view)
            for i in obstacles_view:
                self.viewer.add_geom(i)
            for i in home_regions_view:
                self.viewer.add_geom(i)

        pos_agent1, pos_agent2, pos_goods = self.get_entity_pos()

        self.agent1_trans.set_translation(*map_coor(pos_agent1))
        self.agent2_trans.set_translation(*map_coor(pos_agent2))
        self.goods_trans.set_translation(*map_coor(pos_goods))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()

    def get_entity_pos(self):
        """
        ??????agents????????????goods?????????
        :return:
        """
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if self.map[i][j] == 1:
                    pos_agent1 = (i, j)
                elif self.map[i][j] == 2:
                    pos_agent2 = (i, j)
                elif self.map[i][j] == 3:
                    pos_goods = (i, j)
        return pos_agent1, pos_agent2, pos_goods

    def grasp_goods(self, pos_agent1, pos_agent2, pos_goods):
        """
        ????????????agent???????????????????????????goods
        :param pos_agent1: agent1?????????
        :param pos_agent2: agent2?????????
        :param pos_goods: goods?????????
        :return: ??????????????????True???????????????False
        """

        def judge(_a, _b, _c):
            """
            ??????_a, _b, _c????????????????????????????????????+1???-1
            :param _a:
            :param _b:
            :param _c:
            :return:
            """
            if _b - _a == _c - _b and abs(_c - _b) == 1:
                return True
            return False

        if pos_agent1[0] == pos_agent2[0] and pos_agent2[0] == pos_goods[0]:
            return judge(pos_agent1[1], pos_goods[1], pos_agent2[1])

        # ???????????????????????????????????????2?????????????????????
        # if pos_agent1[1] == pos_agent2[1] and pos_agent2[1] == pos_goods[1]:
        #     return judge(pos_agent1[0], pos_goods[0], pos_agent2[0])

        return False

    def move(self, action_n):
        """
        ???????????????agent??????????????????????????????????????????agent?????????????????????agent??????????????????????????????????????????agent?????????
        :param action_n:
        :return:
        """
        pos_agent1, pos_agent2, pos_goods = self.get_entity_pos()
        if self.grasp_goods(pos_agent1, pos_agent2, pos_goods):
            self.move_grasp(pos_agent1, pos_agent2, pos_goods, action_n)
        else:
            self.move_not_grasp(pos_agent1, pos_agent2, action_n)

    def move_not_grasp(self, pos_agent1, pos_agent2, action_n):
        if action_n[0] == 0 or action_n[1] == 0:
            # ????????????agent?????????????????????????????????
            if action_n[0] == 0 and action_n[1] == 0:
                # ????????????????????????
                return
            if action_n[0] == 0:
                # agent1???????????????
                new_pos_agent2 = (pos_agent2[0] + self.move_delta[action_n[1]][0],
                                  pos_agent2[1] + self.move_delta[action_n[1]][1])
                valid_state2 = self.valid(new_pos_agent2)
                if valid_state2 == 0 or valid_state2 == 2:
                    # agent2????????????????????????goods??????agent1??????agent2????????????
                    return
                else:
                    # agent2??????????????????
                    self.map[new_pos_agent2[0]][new_pos_agent2[1]] = 2
                    self.map[pos_agent2[0]][pos_agent2[1]] = 0
            if action_n[1] == 0:
                # agent2???????????????
                new_pos_agent1 = (pos_agent1[0] + self.move_delta[action_n[0]][0],
                                  pos_agent1[1] + self.move_delta[action_n[0]][1])
                valid_state1 = self.valid(new_pos_agent1)
                if valid_state1 == 0 or valid_state1 == 2:
                    # agent1????????????????????????goods??????agent2??????agent1????????????
                    return
                else:
                    # agent1??????????????????
                    self.map[new_pos_agent1[0]][new_pos_agent1[1]] = 1
                    self.map[pos_agent1[0]][pos_agent1[1]] = 0
        else:
            # ??????agent????????????????????????????????????
            new_pos_agent1 = (pos_agent1[0] + self.move_delta[action_n[0]][0],
                              pos_agent1[1] + self.move_delta[action_n[0]][1])
            new_pos_agent2 = (pos_agent2[0] + self.move_delta[action_n[1]][0],
                              pos_agent2[1] + self.move_delta[action_n[1]][1])
            valid_state1 = self.valid(new_pos_agent1)
            valid_state2 = self.valid(new_pos_agent2)
            if new_pos_agent1 == new_pos_agent2:
                # ??????agent????????????????????????
                return
            if valid_state1 == 0:
                # agent1?????????????????????
                if valid_state2 == 2:
                    # agent2????????????agent1?????????agent2????????????
                    return
                elif valid_state2 == 0:
                    # agent2???????????????
                    return
                else:
                    # agent2???????????????
                    self.map[new_pos_agent2[0]][new_pos_agent2[1]] = 2
                    self.map[pos_agent2[0]][pos_agent2[1]] = 0
            elif valid_state1 == 2:
                # agent1????????????agent2?????????
                if valid_state2 == 0:
                    # agent2?????????????????????
                    return
                elif valid_state2 == 2:
                    # agent2????????????agent1
                    return
                else:
                    # agent2??????????????????agent1???agent2??????????????????
                    self.map[new_pos_agent2[0]][new_pos_agent2[1]] = 2
                    self.map[pos_agent2[0]][pos_agent2[1]] = 0
                    self.map[new_pos_agent1[0]][new_pos_agent1[1]] = 1
                    self.map[pos_agent1[0]][pos_agent1[1]] = 0
            else:
                # agent1??????????????????valid_state1 == 1???
                if valid_state2 == 0:
                    # agent2????????????????????????agent1????????????
                    self.map[new_pos_agent1[0]][new_pos_agent1[1]] = 1
                    self.map[pos_agent1[0]][pos_agent1[1]] = 0
                elif valid_state2 == 1:
                    # agent2??????????????????agent1????????????
                    self.map[new_pos_agent2[0]][new_pos_agent2[1]] = 2
                    self.map[pos_agent2[0]][pos_agent2[1]] = 0
                    self.map[new_pos_agent1[0]][new_pos_agent1[1]] = 1
                    self.map[pos_agent1[0]][pos_agent1[1]] = 0
                else:
                    # agent2???agent1??????
                    self.map[new_pos_agent1[0]][new_pos_agent1[1]] = 1
                    self.map[pos_agent1[0]][pos_agent1[1]] = 0
                    self.map[new_pos_agent2[0]][new_pos_agent2[1]] = 2
                    self.map[pos_agent2[0]][pos_agent2[1]] = 0

    def move_grasp(self, pos_agent1, pos_agent2, pos_goods, action_n):
        """
        ???agent1???agent2??????????????????????????????????????????3????????????
        :param pos_agent1:
        :param pos_agent2:
        :param pos_goods:
        :param action_n:
        :return:
        """
        if action_n[0] != action_n[1]:
            # ????????????agent???????????????????????????????????????????????????move_not_grasp??????
            # self.move_not_grasp(pos_agent1, pos_agent2, action_n)

            # ????????????agent???????????????????????????agent??????
            return
        else:
            # ??????agent???????????????
            if action_n[0] == 0:
                # ????????????????????????
                return
            else:
                # ???????????????????????????

                new_pos_agent1 = (pos_agent1[0] + self.move_delta[action_n[0]][0],
                                  pos_agent1[1] + self.move_delta[action_n[0]][1])
                new_pos_agent2 = (pos_agent2[0] + self.move_delta[action_n[1]][0],
                                  pos_agent2[1] + self.move_delta[action_n[1]][1])
                new_pos_goods = (pos_goods[0] + self.move_delta[action_n[0]][0],
                                 pos_goods[1] + self.move_delta[action_n[0]][1])
                valid_state1 = self.valid(new_pos_agent1)
                valid_state2 = self.valid(new_pos_agent2)
                valid_goods = self.valid(new_pos_goods)

                if action_n[0] == 2 or action_n[0] == 4:
                    # ?????????????????????
                    # ??????????????????????????????????????????????????????????????????????????????goods??????????????????????????????????????????????????????????????????????????????

                    if valid_state1 == 1 or valid_state2 == 1:
                        # ????????????????????????????????????????????????agent???goods????????????
                        # ???????????????
                        self.map[pos_agent1[0]][pos_agent1[1]] = 0
                        self.map[pos_agent2[0]][pos_agent2[1]] = 0
                        # ???????????????
                        self.map[new_pos_agent1[0]][new_pos_agent1[1]] = 1
                        self.map[new_pos_agent2[0]][new_pos_agent2[1]] = 2
                        self.map[new_pos_goods[0]][new_pos_goods[1]] = 3
                    else:
                        # ???????????????????????????????????????????????????
                        return
                else:
                    # ????????????
                    # 3??????????????????????????????
                    # ??????????????????????????????????????????
                    if valid_state1 == 1 and valid_state2 == 1 and valid_goods == 1:
                        # ???????????????
                        self.map[pos_agent1[0]][pos_agent1[1]] = 0
                        self.map[pos_agent2[0]][pos_agent2[1]] = 0
                        self.map[pos_goods[0]][pos_goods[1]] = 0
                        # ???????????????
                        self.map[new_pos_agent1[0]][new_pos_agent1[1]] = 1
                        self.map[new_pos_agent2[0]][new_pos_agent2[1]] = 2
                        self.map[new_pos_goods[0]][new_pos_goods[1]] = 3

    def valid(self, pos):
        """
        ??????pos??????????????????????????????????????????????????????0??????4???
        :param pos:
        :return: ??????????????????????????????1???????????????????????????agent???????????????2????????????????????????????????????????????????goods?????????0
        """
        if pos[0] >= 0 and pos[0] < self.map_size[0] and pos[1] >= 0 and pos[1] < self.map_size[1]:
            if self.map[pos[0]][pos[1]] == 0:
                return 1
            if self.map[pos[0]][pos[1]] == 1 or self.map[pos[0]][pos[1]] == 2:
                return 2
        return 0


if __name__ == '__main__':
    cm = CMOTP_IL()
    cm.__init2__()
    cm.reset2()
    states_n = cm.observation_space.shape
    actions_n = cm.action_space.n
    cm.render3()
    move_str = ["still", 'up', 'right', 'down', 'left']
    import time
    import random

    while True:
        a = random.randint(0, 4)
        b = random.randint(0, 4)
        cm.step_3((a, b))
        print(move_str[a], move_str[b])
        cm.render3()
        time.sleep(1)
    cm.close()
    # env = gym.make('CartPole-v0')
    # for i_episode in range(20):
    #     observation = env.reset()
    #     for t in range(100):
    #         env.render()
    #         print(observation)
    #         action = env.action_space.sample()
    #         observation, reward, done, info = env.step(action)
    #         if done:
    #             print("Episode finished after {} timesteps".format(t + 1))
    #             break
    # env.close()
