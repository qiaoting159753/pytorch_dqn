import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class two_heads_net(nn.Module):
	def __init__(self, action_dims):
		super(two_heads_net, self).__init__()
		#Tensors, with weight and bias.
		self.conv1 = nn.Conv2d(1, 8, 5, stride=2)
		self.conv2 = nn.Conv2d(8, 8, 5, stride=2)
		self.conv3 = nn.Conv2d(8, 8, 5, stride=2)
		self.fc1 = nn.Linear(3128, 256)
		self.fc2 = nn.Linear(256, 256)
		self.fc3_policy = nn.Linear(256, action_dims)
		self.fc3_value = nn.Linear(256, 1)

	def forward(self, x1):
		#Flows
		x1 = (F.relu(self.conv1(x1)))
		x1 = (F.relu(self.conv2(x1)))
		x1 = (F.relu(self.conv3(x1)))
		x1 = x1.view(-1, 3128)
		x1 = F.relu(self.fc1(x1))
		x1 = F.relu(self.fc2(x1))
		x1_policy = F.softmax(self.fc3_policy(x1), dim=1)
		x1_value = self.fc3_value(x1)
		return x1_policy, x1_value

class PPOTrain:
	def __init__(self, state_dims, action_dims):
		self.state_dims = state_dims
		self.network = two_heads_net(action_dims).to(device)
		self.old_network = two_heads_net(action_dims).to(device)
		self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
		self.mse_loss = nn.MSELoss()

	def act(self, obs):
		obs = torch.Tensor(obs).to(device)
		obs = obs.unsqueeze(0)
		probs, value = self.network.forward(obs)
		m = Categorical(probs)
		action = m.sample()
		value = value.detach().cpu().numpy()
		action = action.detach().cpu().numpy()[0]
		return action, value

	def get_gaes(self, rewards, v_preds):
		v_preds_next = v_preds[1:] + [0]
		deltas = [r_t + 0.95 * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
		# calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
		gaes = copy.deepcopy(deltas)
		for t in reversed(range(len(rewards) - 1)):  # is T-1, where T is time step which run policy
			gaes[t] = deltas[t] + 0.98 * gaes[t + 1]
		gaes = np.array(gaes).astype(dtype=np.float32)
		gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-5)
		return gaes

	def train(self, states, actions, rewards, values):
		# Get Gae
		next_observations = states[1:]
		observations = states[0:len(states)-1]

		gaes = self.get_gaes(rewards, values)
		observations = np.reshape(observations, newshape=[-1] + self.state_dims)
		next_observations = np.reshape(next_observations, newshape=[-1] + self.state_dims)
		rewards = np.array(rewards).astype(dtype=np.float32)
		actions = np.array(actions)

		inp = [observations, actions, rewards, gaes, next_observations]
		# train
		for epoch in range(10):
			sample_indices = np.random.randint(low=0, high=gaes.shape[0]-1, size=64)
			sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]
			# Get sampled data
			sampled_obs = sampled_inp[0]
			sampled_actions = sampled_inp[1]
			sampled_rewards = sampled_inp[2]
			sampled_gaes = sampled_inp[3]
			sampled_obs_next = sampled_inp[4]

			# Collected data
			clip_value = 0.2
			sampled_obs = torch.Tensor(sampled_obs).float().to(device)
			sampled_obs_next = torch.Tensor(sampled_obs_next).float().to(device)
			sampled_actions = torch.Tensor(sampled_actions).long().to(device)
			sampled_rewards = torch.Tensor(sampled_rewards).float().to(device)
			sampled_gaes = torch.Tensor(sampled_gaes).float().to(device)

			# Process the observations
			act_probs, sampled_v_preds = self.network(sampled_obs)
			act_probs_next, sampled_v_preds_next = self.network(sampled_obs_next)

			# Get the old log probs
			act_probs_old, _ = self.old_network(sampled_obs)
			act_probs_old = act_probs_old.detach()
			act_dists_old = Categorical(act_probs_old)
			old_logprobs = act_dists_old.log_prob(sampled_actions)

			# Get corresponding new log probs
			act_dists = Categorical(act_probs)
			logprobs = act_dists.log_prob(sampled_actions)

			# Update the policy
			ratios = torch.exp(logprobs - old_logprobs.detach())
			surr1 = ratios * sampled_gaes
			cliped_ratio = torch.clamp(ratios, 1 - clip_value, 1 + clip_value)
			surr2 = cliped_ratio * sampled_gaes

			#Update the value first by TDE
			sampled_v_preds = torch.squeeze(sampled_v_preds)
			sampled_v_preds_next = torch.squeeze(sampled_v_preds_next)
			td = sampled_rewards + 0.95 * sampled_v_preds_next
			loss_vf = self.mse_loss(td, sampled_v_preds)
			#Entropy
			dist_entropy = act_dists.entropy().mean()

			loss = -torch.min(surr1, surr2).mean() + 0.5 * loss_vf - 0.001 * dist_entropy
			self.optimizer.zero_grad()
			loss.mean().backward()
			self.optimizer.step()
		self.old_network.load_state_dict(self.network.state_dict())