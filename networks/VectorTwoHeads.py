import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class Two_Heads_FC(nn.Module):
	def __init__(self, action_dims):
		super(Two_Heads_FC, self).__init__()
		self.fc1 = nn.Linear(33600, 256)
		self.fc2 = nn.Linear(256, 256)
		self.fc3_value = nn.Linear(256, 1)
		self.fc3_policy = nn.Linear(256, action_dims)

	def forward(self, x1):
		x1 = (F.relu(self.fc1(x1)))
		x1 = (F.relu(self.fc2(x1)))
		x1_policy = F.softmax(self.fc3_policy(x1), dim=1)
		x1_value = self.fc3_value(x1)
		return x1_policy, x1_value
