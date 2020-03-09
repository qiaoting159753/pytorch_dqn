import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


class Two_Heads_Net(nn.Module):
	def __init__(self, action_dims):
		super(Two_Heads_Net, self).__init__()
		# Tensors, with weight and bias.
		self.conv1 = nn.Conv2d(1, 8, 5, stride=2)
		self.conv2 = nn.Conv2d(8, 8, 5, stride=2)
		self.conv3 = nn.Conv2d(8, 8, 5, stride=2)
		self.fc1 = nn.Linear(3128, 128)
		self.fc2 = nn.Linear(128, 128)
		self.fc3_policy = nn.Linear(128, action_dims)
		self.fc3_value = nn.Linear(128, 1)

	def forward(self, x1):
		# Flows
		x1 = (F.relu(self.conv1(x1)))
		x1 = (F.relu(self.conv2(x1)))
		x1 = (F.relu(self.conv3(x1)))
		x1 = x1.view(-1, 3128)
		x1 = F.relu(self.fc1(x1))
		x1 = F.relu(self.fc2(x1))
		x1_policy = F.softmax(self.fc3_policy(x1), dim=1)
		x1_value = self.fc3_value(x1)
		return x1_policy, x1_value