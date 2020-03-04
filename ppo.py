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
		#Action
		probs = self.policy(obs)
		m = Categorical(probs)
		action = m.sample()
		action = action.cpu().numpy()
		#value
		value = self.value(obs).cpu()
		value = value.detach().numpy()
		return action, value

	def get_gaes(self, rewards, v_preds, v_preds_next):
		deltas = [r_t + 0.95 * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
		# calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
		gaes = copy.deepcopy(deltas)
		for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
			gaes[t] = gaes[t] + 0.95 * gaes[t + 1]
		gaes = np.array(gaes).astype(dtype=np.float32)
		gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-5)
		return gaes

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
			sample_indices = np.random.randint(low=0, high=observations.shape[0]-1,
											   size=8)  # indices are in [low, high)
			sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
			sampled_obs=sampled_inp[0]
			sampled_actions=sampled_inp[1]
			sampled_rewards=sampled_inp[2]
			sampled_v_preds_next=sampled_inp[3]
			sampled_gaes=sampled_inp[4]
			clip_value = 0.2
			#Evaluate
			sampled_obs = torch.Tensor(sampled_obs).float().to(device)
			sampled_actions = torch.Tensor(sampled_actions).long().to(device)
			sampled_rewards = torch.Tensor(sampled_rewards).float().to(device)
			sampled_v_preds_next = torch.Tensor(sampled_v_preds_next).float().to(device)
			sampled_gaes = torch.Tensor(sampled_gaes).float().to(device)

			act_probs = self.policy(sampled_obs)
			act_probs_old = self.old_policy(sampled_obs)
			act_dists = Categorical(act_probs)
			act_dists_old = Categorical(act_probs_old)
			logprobs = act_dists.log_prob(sampled_actions)
			old_logprobs = act_dists_old.log_prob(sampled_actions)

			sampled_v_preds = self.value(sampled_obs)
			sampled_v_preds = torch.squeeze(sampled_v_preds)

			#Update the value first by TDE
			td = sampled_rewards + 0.95 * sampled_v_preds_next
			loss_vf = self.mse_loss(td, sampled_v_preds)

			# Update the policy
			ratios = torch.exp(logprobs - old_logprobs.detach())
			surr1 = ratios * sampled_gaes
			cliped_ratio = torch.clamp(ratios, 1 - clip_value, 1 + clip_value)
			surr2 = cliped_ratio * sampled_gaes

			#Entropy
			dist_entropy = act_dists.entropy()
			loss = -torch.min(surr1, surr2) - 0.0001 * dist_entropy

			self.actor_optimizer.zero_grad()
			loss.mean().backward()
			self.actor_optimizer.step()

			self.critic_optimizer.zero_grad()
			loss_vf.backward()
			self.critic_optimizer.step()

		self.old_policy.load_state_dict(self.policy.state_dict())
		self.old_value.load_state_dict(self.value.state_dict())