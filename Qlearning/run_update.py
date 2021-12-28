# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 14:43:56 2021

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

@author: User
"""

from maze_env import Maze
from RL_brain import QLearningTable

def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()  #初始位置信息

        while True:
            # fresh env
            env.render()  #环境更新

            # RL choose action based on observation
            action = RL.choose_action(str(observation))  #基于观测值挑选动作

            # RL take action and get next observation and reward
            # action = np.random.choice(state_action[state_action == np.max(state_action)].index)处的返回值
            observation_, reward, done = env.step(action) #进行下一步动作 observation_表示下一个观测值
                  #done判断回合是否结束
            
            # RL learn from this transition 在learn中学习
            RL.learn(str(observation), action, reward, str(observation_)) #完成一次transition

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()
    
if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()