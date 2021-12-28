# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 10:40:17 2021

@author: User
"""



from maze_env import Maze
from RL_brain import SarsaTable


def update():
    for episode in range(100):  #100个回合
        # initial observation
        observation = env.reset()

        # RL choose action based on observation
        action = RL.choose_action(str(observation))  #sarsa根据环境选择action

        while True:
            # fresh env
            env.render()  #环境更新

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))   #采取下一个回合action

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation and action
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()