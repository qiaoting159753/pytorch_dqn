# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import gym
import torch
import torch.nn as nn
import random
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, delta_epsilon):
        super(DQN, self).__init__()

        self.q_value = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, action_dim)
        )

        self.q_value_next = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, action_dim)
        )

        self.delta_epsilon = delta_epsilon
        self.epsilon = 0.01
        self.MseLoss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_value.parameters(), lr=0.001)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        bandit = random.random()
        if bandit < self.epsilon:
            # 1 - Epsilon
            state = torch.from_numpy(state).float().to(device)
            q_values = self.q_value(state)
            value, indices = q_values.max(0)
            action = indices.cpu().detach().numpy()
        else:
            action = random.randint(0, 1)
        return action

    def learn(self, states, actions, rewards, is_terminals):
        next_state = torch.from_numpy(np.asarray(states[1:len(states)])).float().to(device)
        states = torch.from_numpy(np.asarray(states[0:len(states) - 1])).float().to(device)
        rewards = torch.from_numpy(np.asarray(rewards)).float().to(device)
        actions = torch.from_numpy(np.asarray(actions)).long().to(device)
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


def main():
    env = gym.make('CartPole-v1')
    n_latent_var = 64
    delta_epsilon = 0.0001

    agent = DQN(env.reset().shape[0], 2, n_latent_var, delta_epsilon).to(device)

    max_episodes = 10000
    max_steps = 10000
    time_step = 300

    st_rd = []

    for i_episode in range(max_episodes):
        actions = []
        states = []
        rewards = []
        is_terminals = []
        state = env.reset()
        states.append(state)

        for i_step in range(max_steps):
            action = agent.act(state)
            state, reward, done, info = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            is_terminals.append(done)
            env.render()

            if i_step % time_step == 0:
                agent.learn(states, actions, rewards, is_terminals)

            if done:
                break
        # For evaluation
        np_rds = np.asarray(rewards)
        np_rds = sum(np_rds)
        st_rd.append(np_rds)

        if len(st_rd) > 20:
            rdsss = np.asarray(st_rd)
            print(sum(rdsss[len(st_rd) - 20:len(st_rd) - 1]))


if __name__ == '__main__':
    main()


