from __future__ import division
import gym
from ppo import PPOTrain
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    env = gym.make('Assault-v0')
    agent = PPOTrain([3, 250, 160], 7)
    max_episodes = 5000
    max_steps = 600
    st_rd = []

    for i_episode in range(max_episodes):
        actions = []
        states = []
        rewards = []
        values = []
        state = env.reset()
        state = state.reshape(3, 250, 160)
        state = state / 255
        states.append(state)

        total_rewards = 0

        for i_step in range(max_steps):
            action, value = agent.act(state)
            state, reward, done, info = env.step(action[0])
            print(reward)
            total_rewards += reward
            state = state.reshape(3, 250, 160)
            state = state / 255
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value[0, 0])
            if done:
                break
        print("Total: " + str(total_rewards/i_step))
        v_preds_next = values[1:] + [0]
        agent.train(states, actions, rewards, values, v_preds_next)

        # For evaluation
        np_rds = np.asarray(rewards)
        np_rds = sum(np_rds)
        st_rd.append(np_rds)

        if len(st_rd) > 10:
            rdsss = np.asarray(st_rd)
            print(sum(rdsss[len(st_rd) - 10:len(st_rd) - 1])/10)

if __name__ == '__main__':
    main()