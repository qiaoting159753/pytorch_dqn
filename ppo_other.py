import torch.nn.functional as F
import torch.nn as nn
import torch
import gym
import numpy as np
import cv2
import torch.optim as optim
from torch.multiprocessing import Pipe, Process
from collections import deque
from torch.distributions.categorical import Categorical


def make_train_data(reward, done, value, next_value):
	num_step = len(reward)
	discounted_return = np.empty([num_step])

	use_gae = True
	use_standardization = False
	gamma = 0.99
	lam = 0.95
	stable_eps = 1e-30

	# Discounted Return
	if use_gae:
		gae = 0
		for t in range(num_step - 1, -1, -1):
			delta = reward[t] + gamma * \
			        next_value[t] * (1 - done[t]) - value[t]
			gae = delta + gamma * lam * (1 - done[t]) * gae

			discounted_return[t] = gae + value[t]

		# For Actor
		adv = discounted_return - value

	else:
		for t in range(num_step - 1, -1, -1):
			running_add = reward[t] + gamma * next_value[t] * (1 - done[t])
			discounted_return[t] = running_add

		# For Actor
		adv = discounted_return - value

	if use_standardization:
		adv = (adv - adv.mean()) / (adv.std() + stable_eps)

	return discounted_return, adv


class CnnActorCriticNetwork(nn.Module):
	def __init__(self):
		super(CnnActorCriticNetwork, self).__init__()
		self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
		self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
		self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
		self.fc1 = nn.Linear(7 * 7 * 64, 512)
		self.fc2 = nn.Linear(512, 512)

		self.actor = nn.Linear(512, 3)
		self.critic = nn.Linear(512, 1)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = x.view(-1, 7 * 7 * 64)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		policy = self.actor(x)
		value = self.critic(x)
		return policy, value


class CNNActorAgent(object):
	def __init__(
			self,
			num_step,
			gamma=0.99,
			lam=0.95,
			use_gae=True,
			use_cuda=False):
		self.model = CnnActorCriticNetwork()
		self.num_step = num_step
		self.gamma = gamma
		self.lam = lam
		self.use_gae = use_gae
		self.learning_rate = 0.00025
		self.epoch = 3
		self.clip_grad_norm = 0.5
		self.ppo_eps = 0.1
		self.batch_size = 32

		self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

		self.device = torch.device('cuda' if use_cuda else 'cpu')

		self.model = self.model.to(self.device)

	def get_action(self, state):
		state = torch.Tensor(state).to(self.device)
		state = state.float()
		policy, value = self.model(state)
		policy = F.softmax(policy, dim=-1).data.cpu().numpy()
		action = self.random_choice_prob_index(policy)
		return action

	@staticmethod
	def random_choice_prob_index(p, axis=1):
		r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
		return (p.cumsum(axis=axis) > r).argmax(axis=axis)

	def forward_transition(self, state, next_state):
		state = torch.from_numpy(state).to(self.device)
		state = state.float()
		policy, value = self.model(state)
		next_state = torch.from_numpy(next_state).to(self.device)
		next_state = next_state.float()
		_, next_value = self.model(next_state)
		value = value.data.cpu().numpy().squeeze()
		next_value = next_value.data.cpu().numpy().squeeze()
		return value, next_value, policy

	def train_model(self, s_batch, target_batch, y_batch, adv_batch):
		s_batch = torch.FloatTensor(s_batch).to(self.device)
		target_batch = torch.FloatTensor(target_batch).to(self.device)
		y_batch = torch.LongTensor(y_batch).to(self.device)
		adv_batch = torch.FloatTensor(adv_batch).to(self.device)
		sample_range = np.arange(len(s_batch))
		with torch.no_grad():
			policy_old, value_old = self.model(s_batch)
			m_old = Categorical(F.softmax(policy_old, dim=-1))
			log_prob_old = m_old.log_prob(y_batch)

		for i in range(self.epoch):
			np.random.shuffle(sample_range)
			for j in range(int(len(s_batch) / self.batch_size)):
				sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]
				policy, value = self.model(s_batch[sample_idx])
				m = Categorical(F.softmax(policy, dim=-1))
				log_prob = m.log_prob(y_batch[sample_idx])
				ratio = torch.exp(log_prob - log_prob_old[sample_idx])
				surr1 = ratio * adv_batch[sample_idx]
				surr2 = torch.clamp(ratio, 1.0 - self.ppo_eps, 1.0 + self.ppo_eps) * adv_batch[sample_idx]
				actor_loss = -torch.min(surr1, surr2).mean()
				critic_loss = F.mse_loss(value.sum(1), target_batch[sample_idx])
				self.optimizer.zero_grad()
				loss = actor_loss + critic_loss
				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
				self.optimizer.step()


class Environment(Process):
	def __init__(
			self,
			is_render,
			env_idx,
			child_conn):
		super(Environment, self).__init__()
		self.daemon = True
		self.env = gym.make('BreakoutDeterministic-v4')
		self.is_render = is_render
		self.env_idx = env_idx
		self.steps = 0
		self.episode = 0
		self.rall = 0
		self.recent_rlist = deque(maxlen=100)
		self.child_conn = child_conn

		self.history = np.zeros([4, 84, 84])

		self.reset()
		self.lives = self.env.env.ale.lives()

	def run(self):
		super(Environment, self).run()
		while True:
			action = self.child_conn.recv()
			if self.is_render:
				self.env.render()

			_, reward, done, info = self.env.step(action + 1)

			if True:
				if self.lives > info['ale.lives'] and info['ale.lives'] > 0:
					force_done = True
					self.lives = info['ale.lives']
				else:
					force_done = done
			else:
				force_done = done

			if force_done:
				reward = -1

			self.history[:3, :, :] = self.history[1:, :, :]
			self.history[3, :, :] = self.pre_proc(
				self.env.env.ale.getScreenGrayscale().squeeze().astype('float32'))

			self.rall += reward
			self.steps += 1

			if done:
				self.history = self.reset()

			self.child_conn.send(
				[self.history[:, :, :], reward, force_done, done])

	def reset(self):
		self.steps = 0
		self.episode += 1
		self.rall = 0
		self.env.reset()
		self.lives = self.env.env.ale.lives()
		self.get_init_state(
			self.env.env.ale.getScreenGrayscale().squeeze().astype('float32'))
		return self.history[:, :, :]

	def pre_proc(self, X):
		x = cv2.resize(X, (84, 84))
		x *= (1.0 / 255.0)

		return x

	def get_init_state(self, s):
		for i in range(4):
			self.history[i, :, :] = self.pre_proc(s)


if __name__ == '__main__':
	use_cuda = True
	use_gae = True
	is_load_model = False
	is_render = False
	use_standardization = True
	lr_schedule = False
	life_done = True
	use_noisy_net = True

	num_worker = 4

	num_step = 64
	ppo_eps = 0.1
	epoch = 3
	batch_size = 32
	max_step = 1.15e8

	learning_rate = 0.00025

	stable_eps = 1e-30
	epslion = 0.1
	entropy_coef = 0.01
	alpha = 0.99
	gamma = 0.99
	clip_grad_norm = 0.5

	agent = CNNActorAgent(
		num_step,
		gamma,
		use_cuda=use_cuda,
		use_gae=use_gae)

	works = []
	parent_conns = []
	child_conns = []
	for idx in range(num_worker):
		parent_conn, child_conn = Pipe()
		work = Environment(is_render, idx, child_conn)
		work.start()
		works.append(work)
		parent_conns.append(parent_conn)
		child_conns.append(child_conn)

	states = np.zeros([num_worker, 4, 84, 84])

	sample_episode = 0
	sample_rall = 0
	sample_step = 0
	sample_env_idx = 0
	global_step = 0
	recent_prob = deque(maxlen=10)
	score = 0

	while True:
		total_state, total_reward, total_done, total_next_state, total_action = [], [], [], [], []
		global_step += (num_worker * num_step)

		for _ in range(num_step):
			actions = agent.get_action(states)

			for parent_conn, action in zip(parent_conns, actions):
				parent_conn.send(action)

			next_states, rewards, dones, real_dones = [], [], [], []
			for parent_conn in parent_conns:
				s, r, d, rd = parent_conn.recv()
				next_states.append(s)
				rewards.append(r)
				dones.append(d)
				real_dones.append(rd)

			score += rewards[sample_env_idx]
			next_states = np.stack(next_states)
			rewards = np.hstack(rewards)
			dones = np.hstack(dones)
			real_dones = np.hstack(real_dones)

			total_state.append(states)
			total_next_state.append(next_states)
			total_reward.append(rewards)
			total_done.append(dones)
			total_action.append(actions)

			states = next_states[:, :, :, :]

			if real_dones[sample_env_idx]:
				sample_episode += 1
				if sample_episode < 1000:
					print('episodes:', sample_episode, '| score:', score)
					score = 0

		total_state = np.stack(total_state).transpose(
			[1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
		total_next_state = np.stack(total_next_state).transpose(
			[1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
		total_reward = np.stack(total_reward).transpose().reshape([-1])
		total_action = np.stack(total_action).transpose().reshape([-1])
		total_done = np.stack(total_done).transpose().reshape([-1])

		value, next_value, policy = agent.forward_transition(
			total_state, total_next_state)
		total_target = []
		total_adv = []
		for idx in range(num_worker):
			target, adv = make_train_data(total_reward[idx * num_step:(idx + 1) * num_step],
			                              total_done[idx * num_step:(idx + 1) * num_step],
			                              value[idx * num_step:(idx + 1) * num_step],
			                              next_value[idx * num_step:(idx + 1) * num_step])
			# print(target.shape)
			total_target.append(target)
			total_adv.append(adv)

		print('training')
		agent.train_model(
			total_state,
			np.hstack(total_target),
			total_action,
			np.hstack(total_adv))
