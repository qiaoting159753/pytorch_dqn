import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class policy_net(nn.Module):
	def __init__(self, action_dims):
		super(policy_net, self).__init__()
		#Tensors, with weight and bias.
		self.conv1 = nn.Conv2d(3, 8, 5)
		self.pool1 = nn.MaxPool2d(2, 2, 1)
		self.conv2 = nn.Conv2d(8, 8, 5)
		self.pool2 = nn.MaxPool2d(2, 2, 1)
		self.conv3 = nn.Conv2d(8, 8, 5)
		self.pool3 = nn.MaxPool2d(2, 2, 1)
		self.fc1 = nn.Linear(4176, 512)
		self.fc2 = nn.Linear(512, 512)
		self.fc3 = nn.Linear(512, action_dims)

	def forward(self, x1):
		#Flows
		x1 = self.pool1(F.relu(self.conv1(x1)))
		x1 = self.pool2(F.relu(self.conv2(x1)))
		x1 = self.pool3(F.relu(self.conv3(x1)))
		x1 = x1.view(-1, 4176)
		x1 = F.relu(self.fc1(x1))
		x1 = F.relu(self.fc2(x1))
		x1 = self.fc3(x1)
		x1 = F.softmax(x1, dim=1)
		return x1

class value_net(nn.Module):
	def __init__(self):
		super(value_net, self).__init__()
		#Tensors, with weight and bias.
		self.conv1 = nn.Conv2d(3, 8, 5)
		self.pool1 = nn.MaxPool2d(2, 2, 1)
		self.conv2 = nn.Conv2d(8, 8, 5)
		self.pool2 = nn.MaxPool2d(2, 2, 1)
		self.conv3 = nn.Conv2d(8, 8, 5)
		self.pool3 = nn.MaxPool2d(2, 2, 1)
		self.fc1 = nn.Linear(4176, 512)
		self.fc2 = nn.Linear(512, 512)
		self.fc3 = nn.Linear(512, 1)

	def forward(self, x1):
		#Flows
		x1 = self.pool1(F.relu(self.conv1(x1)))
		x1 = self.pool2(F.relu(self.conv2(x1)))
		x1 = self.pool3(F.relu(self.conv3(x1)))
		x1 = x1.view(-1, 4176)
		x1 = F.relu(self.fc1(x1))
		x1 = F.relu(self.fc2(x1))
		x1 = (self.fc3(x1))
		return x1

class PPOTrain:
	def __init__(self, state_dims, action_dims):
		self.state_dims = state_dims
		self.policy = policy_net(action_dims).to(device)
		self.value = value_net().to(device)
		self.old_policy = policy_net(action_dims).to(device)
		self.old_value = value_net().to(device)
		self.actor_optimizer = optim.Adam(self.policy.parameters(), lr=0.0001)
		self.critic_optimizer = optim.Adam(self.value.parameters(), lr=0.0001)
		self.mse_loss = nn.MSELoss()

	def get_value(self, next):
		obs = torch.Tensor(next).to(device)
		values = self.value(obs)
		return values.cpu().detach().numpy()

	def act(self, obs):
		new_obs = np.asarray(obs)
		new_obs = np.expand_dims(new_obs, axis=0)
		obs = torch.Tensor(new_obs).to(device)
		# Action
		probs = self.policy(obs)
		m = Categorical(probs)
		action = m.sample()
		action = action.cpu().numpy()
		# value
		value = self.value(obs).cpu()
		value = value.detach().numpy()
		return action, value

	def train(self, states, actions, rewards, values, next_values):
		# Get Gae
		gaes = self.get_gaes(rewards, values, next_values)
		observations = np.reshape(states, newshape=[-1] + self.state_dims)
		actions = np.array(actions).astype(dtype=np.int32)
		rewards = np.array(rewards).astype(dtype=np.float32)
		v_preds_next = np.array(next_values).astype(dtype=np.float32)

		inp = [observations, actions, rewards, v_preds_next, gaes]
		# train
		for epoch in range(5):
