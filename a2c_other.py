import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn as nn
import gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TwoHeadNetwork(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(TwoHeadNetwork, self).__init__()
		self.policy1 = nn.Linear(input_dim, 256)
		self.policy2 = nn.Linear(256, output_dim)
		self.value1 = nn.Linear(input_dim, 256)
		self.value2 = nn.Linear(256, 1)

	def forward(self, state):
		logits = F.relu(self.policy1(state))
		logits = self.policy2(logits)
		value = F.relu(self.value1(state))
		value = self.value2(value)
		return logits, value


class A2CAgent():
	def __init__(self, env, gamma, lr):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.env = env
		self.obs_dim = env.observation_space.shape[0]
		self.action_dim = env.action_space.n

		self.gamma = gamma
		self.lr = lr

		self.model = TwoHeadNetwork(self.obs_dim, self.action_dim).to(device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

	def get_action(self, state):
		state = torch.FloatTensor(state).to(self.device)
		logits, _ = self.model.forward(state)
		dist = F.softmax(logits, dim=0)
		probs = Categorical(dist)
		return probs.sample().cpu().detach().item()

	def update(self, trajectory):
		states = torch.FloatTensor([sars[0] for sars in trajectory]).to(self.device)
		actions = torch.LongTensor([sars[1] for sars in trajectory]).view(-1, 1).to(self.device)
		rewards = torch.FloatTensor([sars[2] for sars in trajectory]).to(self.device)

		# compute discounted rewards
		discounted_rewards = [
			torch.sum(torch.FloatTensor([self.gamma ** i for i in range(rewards[j:].size(0))]).to(device) * rewards[j:]) for j in range(rewards.size(0))]  # sorry, not the most readable code.

		# V_t = r + gamma * V_t+1
		value_targets = rewards.view(-1, 1) + torch.FloatTensor(discounted_rewards).view(-1, 1).to(self.device)

		logits, values = self.model.forward(states)
		dists = F.softmax(logits, dim=1)
		probs = Categorical(dists)

		# compute value loss
		value_loss = F.mse_loss(values, value_targets.detach())

		# compute entropy bonus
		entropy = []
		for dist in dists:
			entropy.append(-torch.sum(dist.mean() * torch.log(dist)))
		entropy = torch.stack(entropy).sum()

		# compute policy loss
		# A = V_t - V
		advantage = value_targets - values
		# log * A
		policy_loss = -probs.log_prob(actions.view(actions.size(0))).view(-1, 1) * advantage.detach()
		policy_loss = policy_loss.mean()

		total_loss = policy_loss + value_loss - 0.001 * entropy
		self.optimizer.zero_grad()
		total_loss.backward()
		self.optimizer.step()


env = gym.make("CartPole-v0")
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
MAX_EPISODE = 2500
MAX_STEPS = 500

lr = 1e-4
gamma = 0.99

agent = A2CAgent(env, gamma, lr)


def run():
	for episode in range(MAX_EPISODE):
		state = env.reset()
		trajectory = []  # [[s, a, r, s', done], [], ...]
		episode_reward = 0
		for steps in range(MAX_STEPS):
			action = agent.get_action(state)
			next_state, reward, done, _ = env.step(action)
			trajectory.append([state, action, reward, next_state, done])
			episode_reward += reward

			if done:
				break

			state = next_state
		if episode % 10 == 0:
			print("Episode " + str(episode) + ": " + str(episode_reward))
		agent.update(trajectory)


run()
