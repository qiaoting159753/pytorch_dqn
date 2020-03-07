from __future__ import division
import gym
from a2c import A2C
import torch
import numpy as np
from skimage.color import rgb2grey

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def preprocess(state):
    state = rgb2grey(state)
    state = state.reshape([1, 210, 160])
    state /= 255
    return state

def main():
    env = gym.make('Pong-v0')
    agent = A2C([1, 210, 160], 6)
    max_episodes = 5000
    max_steps = 1000
    timesteps = 100

    st_rd = []

    for i_episode in range(max_episodes):
        actions = []
        states = []
        rewards = []
        values = []
        logs = []
        entropies = []

        state = env.reset()
        state = preprocess(state)
        states.append(state)
        total_rewards = 0

        for i_step in range(max_steps):
            #value is detached.
            action, value, log, entropy = agent.act(state)
            state, reward, done, info = env.step(action)
            state = preprocess(state)
            total_rewards += reward
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value[0, 0])
            logs.append(log)
            entropies.append(entropy)

            if i_step % timesteps == 0:
                agent.train(states, rewards, values, logs, entropies)
                actions = []
                states = []
                rewards = []
                values = []
                logs = []
                entropies = []
                states.append(state)

            if done:
                break
        agent.train(states, rewards, values, logs, entropies)
        #For evaluation
        np_rds = np.asarray(rewards)
        np_rds = sum(np_rds)
        st_rd.append(np_rds)

        if len(st_rd) > 10:
            rdsss = np.asarray(st_rd)
            print(sum(rdsss[len(st_rd) - 10:len(st_rd) - 1])/10)

if __name__ == '__main__':
    main()