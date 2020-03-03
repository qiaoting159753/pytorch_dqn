from __future__ import division
import gym
from dqn import DQN
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    env = gym.make('Assault-v0')
    n_latent_var = 64
    delta_epsilon = 0.0001

    agent = DQN(env.reset().shape[0], 3, n_latent_var, delta_epsilon).to(device)

    max_episodes = 5000
    max_steps = 500
    time_steps = 100

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

            if i_step % time_steps == 0:
                agent.learn(states, actions, rewards, is_terminals)

            if done:
                break
        agent.learn(states, actions, rewards, is_terminals)

        # For evaluation
        np_rds = np.asarray(rewards)
        np_rds = sum(np_rds)
        st_rd.append(np_rds)

        if len(st_rd) > 50:
            rdsss = np.asarray(st_rd)
            print(sum(rdsss[len(st_rd) - 50:len(st_rd) - 1])/50)
            #if len(st_rd) % 100 == 0:
                #np.save("C:/Users/tingq/Desktop/data/" + str(len(st_rd)) + ".npy", rdsss)


if __name__ == '__main__':
    main()