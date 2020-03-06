import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Two_Heads_Net(nn.Module):
	def __init__(self, action_dims):
		super(Two_Heads_Net, self).__init__()
		#Tensors, with weight and bias.
		self.conv1 = nn.Conv2d(3, 8, 5, stride=2)
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

class A2C:
	def __init__(self, state_dims, action_dims):
		self.state_dims = state_dims
		self.network = Two_Heads_Net(action_dims).to(device)
		self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
		self.mse_loss = nn.MSELoss()

	def act(self, obs):
		obs = torch.Tensor(obs).to(device)
		obs = obs.unsqueeze(0)
		probs, value = self.network.forward(obs)
		dist = Categorical(probs)
		action = dist.sample()
		value = value.detach().cpu().numpy()
		return (action, value, dist.log_prob(action), dist.entropy())

	def get_gaes(self, rewards, v_preds):
		v_preds_next = v_preds[1:] + [0]
		deltas = [r_t + 0.95 * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
		# calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
		gaes = (deltas)
		for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
			gaes[t] = gaes[t] + 0.95 * gaes[t + 1]
		gaes = np.array(gaes).astype(dtype=np.float32)
		gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-5)
		return gaes

	def a2c_loss(self, final_value, entropy):
		# weight the deviation of the predicted value (of the state) from the
		# actual reward (=advantage) with the negative log probability of the action taken
		policy_loss = (-self.log_probs * advantage.detach()).mean()

		# the value loss weights the squared difference between the actual
		# and predicted rewards
		value_loss = advantage.pow(2).mean()

		# return the a2c loss
		# which is the sum of the actor (policy) and critic (advantage) losses
		# due to the fact that batches can be shorter (e.g. if an env is finished already)
		# MEAN is used instead of SUM
		loss = policy_loss + 0.5 * value_loss - 0.001 * entropy

		return loss

	def train(self,states, actions, rewards, values, logs, entropies):
		gaes = self.get_gaes(rewards, values)
		policy_loss = (-logs * gaes)
		value_loss =

		print "Training"