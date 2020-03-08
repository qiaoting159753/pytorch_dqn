from __future__ import division
import gym
from ppo import PPOTrain
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
    env = gym.make('Boxing-v0')
    agent = PPOTrain([1, 210, 160], 18)
    max_episodes = 5000
    max_steps = 1000
    timesteps = 64

    st_rd = []
    total_steps = 0

    for i_episode in range(max_episodes):
        actions = []
        states = []
        rewards = []
        values = []

        state = env.reset()
        state = preprocess(state)
        states.append(state)
        total_rewards = 0

        for i_step in range(max_steps):
            #value is detached.
            action, value = agent.act(state)
            state, reward, done, info = env.step(action)
            state = preprocess(state)
            total_rewards += reward

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value[0, 0])

            if i_step % timesteps == 0 and i_step > 10:
                agent.train(states, actions, rewards, values)
                actions = []
                states = []
                rewards = []
                values = []
                states.append(state)

            if done:
                break

        total_steps += i_step
        agent.train(states, actions, rewards, values)
        #For evaluation
        st_rd.append(total_rewards)
        print(total_rewards)
        if len(st_rd) > 10:
            rdsss = np.asarray(st_rd)
            print(sum(rdsss[len(st_rd) - 10:len(st_rd) - 1])/10)

if __name__ == '__main__':
    main()