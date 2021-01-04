#!/usr/bin/env python
# coding: utf-8

# In[1]:


from environment import MountainCar


# In[2]:


import numpy as np


# In[3]:


def generate_q(s,a,w,b):
    #state, action , weight, bias
    value = 0
    c = w[:, a]
    for i,k in s.items():
        value += c[i]*k
    value += b
    return value


# In[4]:


def generate_q_max(state_next, action_set, w, b):
    q_set = []
    for a in action_set:
        single_q = generate_q(state_next, a, w, b)
        q_set.append(single_q)
        
    q_max = max(q_set)
    return q_max
        


# In[5]:


from numpy import random


# In[6]:


def action_select(s, action_set, w, b, e):
    #epsilon-greedy action selection 
    q_set = []
    if random.random() >= e:
        for i in range(len(action_set)):
            single_q = generate_q(s,i,w,b)
            q_set.append(single_q)
        action = q_set.index(max(q_set))
        #choose the max-q-action
        
    else:
        action = random.choice(action_set)
        #choose randomly
    return action
            


# In[7]:


def single_episode(mode, action_set, w, b, e, lr, gamma, max_iteration, returns):

    s = mode.transform(mode.state)
    #initialize
    r_t = 0
    step = 0
    done = False
        
    while (done == False):
        a = action_select(s, action_set, w, b, e)
        s_next, reward, done = mode.step(a)
        q = generate_q(s,a,w,b)
        r_t += reward
        q_max = generate_q_max(s_next, action_set, w, b)
        for i,k in s.items():
            w[i][a] = w[i][a] - (lr*(q - (reward + gamma * q_max)))*k
            
        b -= lr*(q - (reward + gamma * q_max))
        s = s_next
        step += 1
        
        if step == max_iteration:
            #reach the max_iteration
            #there two conditions to end: 1.done == True，2.reach the max_iteration
            break
            
    returns.append(r_t)
    
    mode.reset()
    
    return returns, w, b


# In[8]:


def Q_learning(episode, mode, action_set, w, b, e, lr, gamma, max_iteration):
    returns = []
    for i in range(episode):
        returns, w, b = single_episode(mode, action_set, w, b, e, lr, gamma, max_iteration, returns)
        
    return returns, w, b


# In[9]:


def generate_weight_file(file_path, w, b):
    f1 = open(file_path,'w')
    f1.write(str(b))
    f1.write('\n')
    for i in range(len(w)):
        for k in range(len(w[0])):
            f1.write(str(w[i,k]))
            f1.write('\n')
                        
    f1.close()
            


# In[10]:


def generate_returns_flie(file_path, r):
    f2 = open(file_path,'w')
    for i in range(len(r)):
        f2.write(str(r[i]))
        f2.write('\n')
        
    f2.close()


# In[ ]:


import sys


# In[ ]:


if __name__ == "__main__":
    
    mode = MountainCar(sys.argv[1])
    weight_out = sys.argv[2]
    returns_out = sys.argv[3]
    episodes = int(sys.argv[4])
    max_iterations = int(sys.argv[5])
    epsilon = float(sys.argv[6])
    gamma = float(sys.argv[7])
    learning_rate = float(sys.argv[8])
    
    #initial weight and bias
    action_set = [0,1,2]
    weight_initial = np.zeros([mode.state_space, len(action_set)], dtype=float)
    bias_initial = 0
    
    #after the q_learning update
    returns,weight, bias = Q_learning(episodes, mode, action_set, weight_initial, bias_initial, epsilon, learning_rate, gamma, max_iterations)
    #write file
    generate_weight_file(weight_out, weight, bias)
    generate_returns_flie(returns_out, returns)

