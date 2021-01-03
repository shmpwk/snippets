#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import os 
import datetime

class SimpleBuffer:
    def __init__(self, data_length, buffer_size, rgb_shape, depth_shape, state_shape, robot_state_shape, device=torch.device('cuda')):
        # GPU上に保存するデータ．
        self.states = torch.empty((buffer_size, state_shape), dtype=torch.float, device=device)
        self.robot_states = torch.empty((buffer_size, robot_state_shape), dtype=torch.float, device=device)
        self.rgb = torch.empty((buffer_size, rgb_shape), dtype=torch.float, device=device)
        self.depth = torch.empty((buffer_size, depth_shape), dtype=torch.float, device=device)
        #self.tmp_states = torch.empty((data_length, state_shape), dtype=torch.float, device=device)
        # 次にデータを挿入するインデックス．
        self._p = 0
        self.data_length = data_length
        self.buffer_size = buffer_size

    def append(self, rgb, depth, state, robot_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.robot_states[self._p].copy_(torch.from_numpy(robot_state))
        self.rgb[self._p].copy_(torch.from_numpy(rgb))
        self.depth[self._p].copy_(torch.from_numpy(depth))
        self._p = (self._p + 1) % self.buffer_size

    def save(self):
        """
        torch.save({
            'state': self.states.clone().cpu(),
        }, path)
        """
        now = datetime.datetime.now()  
        robot_path = 'data/robot_data/robot_' + now.strftime('%Y%m%d_%H%M%S') + '.pth'
        robot_state_path = 'data/robot_state_data/robot_' + now.strftime('%Y%m%d_%H%M%S') + '.pth'
        rgb_path = 'data/rgb_data/rgb_' + now.strftime('%Y%m%d_%H%M%S') + '.pth'
        depth_path = 'data/depth_data/depth_' + now.strftime('%Y%m%d_%H%M%S') + '.pth'
        # GPU save
        ## Save whole model
        torch.save(self.states, robot_path)
        torch.save(self.robot_states, robot_state_path)
        torch.save(self.rgb, rgb_path)
        torch.save(self.depth, depth_path)
        # CPU save
        #torch.save(self.model.to('cpu').state_dict(), model_path)
 
    def get(self):
        assert self._p == 0, 'Buffer needs to be full before training.'
        return self.rgb, self.depth, self.states, self.robot_states

    def delete(self):
        self._p = self._p - self.data_length
 
    # path should be set by parser
    def load(self, rgb_path, depth_path, robot_path):
        # learn GPU, load GPU
        rgb = torch.load(rgb_path)
        depth = torch.load(depth_path)
        data = torch.load(robot_path)
        # learn CPU, load GPU
        #self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        return rgb, depth, data

"""
For PPO buffer
ここでは，状態・行動・即時報酬・終了シグナル・確率密度の対数をロールアウト1回分保存することとします．このとき，状態のみ1つ分多く保存することに注意します(GAEの計算では，1ステップ先の状態価値を計算する必要があるので)．

class RolloutBuffer:
    def __init__(self, buffer_size, state_shape, action_shape, device=torch.device('cuda')):

        # GPU上に保存するデータ．
        self.states = torch.empty((buffer_size + 1, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty((buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty((buffer_size, 1), dtype=torch.float, device=device)

        # 次にデータを挿入するインデックス．
        self._p = 0
        # バッファのサイズ．
        self.buffer_size = buffer_size

    def append(self, state, action, reward, done, log_pi):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self._p = (self._p + 1) % self.buffer_size

    def append_last_state(self, last_state):
        assert self._p == 0, 'Buffer needs to be full before appending last_state.'
        self.states[self.buffer_size].copy_(torch.from_numpy(last_state))
    
    def get(self):
        assert self._p == 0, 'Buffer needs to be full before training.'
        return self.states, self.actions, self.rewards, self.dones, self.log_pis

class SerializedBuffer:

    def __init__(self, path, device=torch.device('cuda')):
        tmp = torch.load(path)

        self.buffer_size = self._n = tmp['state'].size(0)
        self.device = device

        self.states = tmp['state'].clone().to(self.device)
        self.actions = tmp['action'].clone().to(self.device)
        self.rewards = tmp['reward'].clone().to(self.device)
        self.dones = tmp['done'].clone().to(self.device)
        self.next_states = tmp['next_state'].clone().to(self.device)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes]
        )


class Buffer(SerializedBuffer):

    def __init__(self, buffer_size, state_shape, action_shape, device=torch.device('cuda')):
        self._p = 0
        self._n = 0
        self.buffer_size = buffer_size
        self.device = device

        self.states = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty((buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def save(self, path):
        torch.save({
            'state': self.states.clone().cpu(),
            'action': self.actions.clone().cpu(),
            'reward': self.rewards.clone().cpu(),
            'done': self.dones.clone().cpu(),
            'next_state': self.next_states.clone().cpu(),
        }, path)
"""
