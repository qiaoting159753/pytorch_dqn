import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


class A2C:
	def __init__(self, state_dims, action_dims):
		self.state_dims = state_dims
		self.network = Two_Heads_Net(action_dims).to(device)
		self.optimizer = optim.Adam(self.network.parameters(), lr=0.0001)

	def act(self, obs):
		obs = torch.Tensor(obs).to(device)
		obs = obs.unsqueeze(0)
		probs, value = self.network.forward(obs)
		dist = Categorical(probs)
		action = dist.sample()
		log = dist.log_prob(action)
		action = action.cpu().numpy()[0]
		value = value.detach().cpu().numpy()
		return action, value, log, dist.entropy()

	def get_gaes(self, rewards, v_preds):
		v_preds_next = v_preds[1:] + [0]
		deltas = [r_t + 0.95 * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
		# calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
		gaes = copy.deepcopy(deltas)
		for t in reversed(range(len(rewards) - 1)):  # is T-1, where T is time step which run policy
			gaes[t] = deltas[t] + 0.98 * gaes[t + 1]
			# Because the value will be reduced later.
			gaes[t] += v_preds[t]

		gaes = np.array(gaes).astype(dtype=np.float32)
		gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-5)
		return gaes

	def train(self, states, rewards, values, logs, entropies):
		gaes = self.get_gaes(rewards, values)
		gaes = torch.from_numpy(np.asarray(gaes)).float().to(device)
		gaes = gaes.unsqueeze(1)
		states = torch.from_numpy(np.asarray(states[0:len(states) - 1])).float().to(device)
		prob_grad, values_grad = self.network(states)

		advantage = gaes - values_grad
		value_loss = advantage.pow(2).mean()

		logs = torch.stack(logs)
		policy_loss = (logs * advantage.detach()).mean()

		entropies = torch.stack(entropies).mean()

		loss = -policy_loss + 0.5 * value_loss - 0.001 * entropies

		self.optimizer.zero_grad()
		loss.backward(retain_graph=False)
		nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
		self.optimizer.step()
