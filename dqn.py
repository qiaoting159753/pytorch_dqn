# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Q_Net(nn.Module):
	def __init__(self):
		super(Q_Net, self).__init__()
		# Tensors, with weight and bias.
		self.conv1 = nn.Conv2d(3, 8, 5)
		self.pool1 = nn.MaxPool2d(2, 2, 1)
		self.conv2 = nn.Conv2d(8, 8, 5)
		self.pool2 = nn.MaxPool2d(2, 2, 1)
		self.conv3 = nn.Conv2d(8, 8, 5)
		self.pool3 = nn.MaxPool2d(2, 2, 1)
		self.fc1 = nn.Linear(4176, 512)
		self.fc2 = nn.Linear(512, 512)
		self.fc3 = nn.Linear(512, 7)

	def forward(self, x1):
		# Flows
		x1 = self.pool1(F.relu(self.conv1(x1)))
		x1 = self.pool2(F.relu(self.conv2(x1)))
		x1 = self.pool3(F.relu(self.conv3(x1)))
		x1 = x1.view(-1, 4176)
		x1 = F.relu(self.fc1(x1))
		x1 = F.relu(self.fc2(x1))
		x1 = self.fc3(x1)
		return x1


class DQN():
	def __init__(self, delta_epsilon=0.002):
		self.q_value = Q_Net().to(device)
		self.q_value_next = Q_Net().to(device)
		self.delta_epsilon = delta_epsilon
		self.epsilon = 0.01
		self.MseLoss = nn.MSELoss()
		self.optimizer = torch.optim.Adam(self.q_value.parameters(), lr=0.001)

	def act(self, state):
		bandit = random.random()
		if bandit < self.epsilon:
			# 1 - Epsilon
			state = np.asarray(state)
			state = np.expand_dims(state, axis=0)
			state = torch.from_numpy(state).float().to(device)
			q_values = self.q_value(state)
			value, indices = q_values.max(1)
			action = indices.cpu().detach().numpy()
		else:
			action = random.randint(0, 6)
		return action

	def learn(self, states, actions, rewards):
		next_state = torch.from_numpy(np.asarray(states[1:len(states)])).float().to(device)
		states = torch.from_numpy(np.asarray(states[0:len(states) - 1])).float().to(device)
		rewards = torch.from_numpy(np.asarray(rewards)).float().to(device)
		actions = torch.from_numpy(np.asarray(actions, dtype=np.uint8)).long().to(device)
		q_next = self.q_value_next(next_state)
		q_curr = self.q_value(states)
		q_next_value, q_next_indices = q_next.max(1)

		td = rewards + 0.95 * q_next_value

		loss = self.MseLoss(q_curr.gather(1, actions.unsqueeze(1)).squeeze(), td)
		self.optimizer.zero_grad()
		loss.mean().backward()
		self.optimizer.step()

		self.q_value_next.load_state_dict(self.q_value.state_dict())
		self.epsilon += self.delta_epsilon

