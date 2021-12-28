# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 09:51:00 2021

@author: User
"""

import numpy as np
import pandas as pd
import time #设置agent移动速度

np.random.seed(2) #产生一组伪随机数列

#global variables
N_STATES=6 #number of states
ACTIONS=['left', 'right'] #variable actions
EPSILON=0.9 #greedy police
ALPHA=0.1 #learning rate
LAMBDA=0.9 #discount factor
MAX_EPISODES= 13 #maximum episodes
FRESH_TIME =0.01 #fresh time for one move走一步花0.3s

def build_q_table(n_states, actions):
    table = pd.DataFrame(            
        np.zeros((n_states, len(actions))),   #全0初始化
        columns=actions,
    )      #利用pandas创建一个表格
   # print(table) 
    return table

def choose_action(state, q_table):
    state_actions=q_table.iloc[state, :]    #找到q_table中state对应行赋值给state_actions
    if (np.random.uniform()>EPSILON) or (state_actions.all() ==0):   #10%的情况下随机选择动作或者初始全为0时也随机选择动作
        action_name = np.random.choice(ACTIONS)      
    else: 
        action_name = ACTIONS[state_actions.argmax()]     #90%的情况下选择q值最大的action
    return action_name

def get_env_feedback(S, A):
    if A =='right':
        if S == N_STATES -2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S+1
            R =0
    else:
        R = 0
        if S ==0:
            S_ = S
        else:
            S_ = S-1
    return S_, R

def update_env(S, episode, step_counter):
    env_list = ['-']*(N_STATES-1)+['T']
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction),end='')
        time.sleep(2)
        print('\r                       ',end='')
    else:
        env_list[S]='o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction),end='')
        time.sleep(FRESH_TIME)
    
    
def rl():
     q_table = build_q_table(N_STATES,ACTIONS)
     for episode in range(MAX_EPISODES):
         step_counter = 0
         S = 0
         is_terminated = False  #终止符
         update_env(S, episode, step_counter)
         while not is_terminated:
         
             A = choose_action(S, q_table)
             S_, R = get_env_feedback(S, A)
             q_predict = q_table.loc[S, A]
             if S_ !='terminal':
                 q_target = R + LAMBDA * q_table.iloc[S_, :].max()
             else:
                 q_target = R
                 is_terminated = True  #跳出while循环 进入下一个episode
                
             q_table.loc[S, A] += ALPHA * (q_target - q_predict)
             S = S_
            
             update_env(S, episode, step_counter+1)
             
             step_counter +=1
     return q_table
         
if __name__== "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
    