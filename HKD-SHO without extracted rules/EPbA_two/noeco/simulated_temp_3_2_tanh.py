# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 08:14:07 2021

@author: mingm
"""
# %% light intensity
import torch
import matplotlib
# matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{mathptmx}']
# process = subprocess.Popen(command, stdout=tempFile, shell=True)

import numpy as np
from astropy import modeling

from configurations import *

# %%
device=torch.device("cpu")
# device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# fig,ax = plt.subplots()

# # m = modeling.models.Gaussian1D(amplitude=600, mean=12*60/5, stddev=3*60/5)
# # x = np.linspace(0, 288, 288)
# # data = m(x)
# # data = data + 5 * np.random.random(x.size)


# m2 = modeling.models.Gaussian1D(amplitude=600, mean=12, stddev=3)
# x2 = np.linspace(0, 24, 288)
# data2 = m2(x2)
# data2 = data2 + 5 * np.random.random(x2.size)
# # data -= data.min()
# # plt.grid()
# # plt.plot(x, data,color='b')
# plt.plot(x2, data2,linewidth=4)



# # plt.xticks(np.arange(0, 288, step=1)) 

# plt.setp(ax.get_xticklabels(), fontsize=20)
# plt.setp(ax.get_yticklabels(), fontsize=20)

# l=[0,4,8,12,16,20,24]
# labels = [0,48,96,144,192,240,288]

# ax.set_xticks(l)



# # plt.legend(fontsize=20)
# # plt.xticks(x2, np.arange(288,step=100))
# ax.set_xticklabels(labels)

# ax.set_xlabel('step',fontsize=20)
# ax.set_ylabel('$\displaystyle x_{{le}_t}$',fontsize=20)
# # plt.grid()

# # plt.savefig('figs/simulated_x_le_t.eps')

# plt.show()

# # print(data2)

# %% temperature
import numpy as np # linear algebra

# the high temperature for the day is 27°C, the low temperature is 11°C at 4am
# midline value: (27+11)/2=19
# D(4)=12  (4,12), (28,12)
# D(16)=30 (16,30)
# T=24
# A=27-19=-8  2*pi/B=24 -> B=pi/12  D=4  C=19 
# *********************************************************************************************************
# *********************************************************************************************************
# *********************************************************************************************************

# min=12
# max=26
# y=19
# (4,12)
# (28,12)
# (16,26)
# T=24

def temp_simulator():
    np.random.seed(0)
    fig,ax = plt.subplots()
    
    A=-7
    B=np.pi/12
    C=19
    D=4
        
    x = np.linspace(0, 24, 288)
    data1=A*np.cos(B*(x-D))+C
    data1 = data1 
    
    x_temp_t=torch.from_numpy(data1).reshape((1,-1))
    
    x_temp_t=x_temp_t.to(device)
    
    return x_temp_t


# %%
# np.random.seed(0)
# fig,ax = plt.subplots()

# A=-8
# B=np.pi/12
# C=19
# D=4
    
# x = np.linspace(0, 24, 288)
# data1=A*np.cos(B*(x-D))+C
# data1 = data1 + 0.5*np.random.normal(size=x.size)
    
# import matplotlib.pyplot as plt
# plt.plot(x,data1)
# plt.grid()
# ax.set_xlabel('step',fontsize=20)
# ax.set_ylabel('$\displaystyle x_{{te}_t}$',fontsize=20)
# plt.show()

# # %%
# # outdoor air quality
# import numpy as np # linear algebra

# # assumption: two sinusoidal functions are combined together to get the final air quality model
# # the first sinusoidal line: the maximum value: 500ppm, at 8pm (8,500)(20,500); the minimum value: 410ppm, at 5pm (5,410)(17,410)
#     # the middle value: (500+410)/2=455
# # the second sinusoidal line: the maximum value: 480ppm, at 17pm (17,480)(25,480); the minimum value: 420ppm,at 19pm (19,420)(27,420);
#     # the middle value: (480+420)/2=450



# A1=-(500-455)
# A2=(480-450)

# B1=np.pi/3
# B2=np.pi/2

# D1=5
# D2=17

# C1=455
# C2=450

  
# x = np.linspace(0, 24, 288)
# data1=A1*np.cos(B1*(x-D1))+C1
# data2=A2*np.cos(B2*(x-D2))+C2
# # data1 = data1 + 5*np.random.random(x.size)
# # data2 = data2 + 5*np.random.random(x.size)

# # for i in range(len(x)):
    
# #     if abs(data1[i]-data2[i])<0.1:
# #         print(i)
# #     if i==217:
# #         print(data1[i])
# #         print(data2[i])

# data=None



# import matplotlib.pyplot as plt
# plt.plot(x,data1,label='1')
# plt.plot(x,data2,label='2')
# plt.grid()
# plt.legend()
# plt.show()
# # weather_temp = weather[["Formatted Date","Apparent Temperature (C)"]]
# # plt.plot(weather_temp['Apparent Temperature (C)'])

# # %%
# from collections import deque
# import random

# ReplayMemorySize=1000000
# replayMemory=deque(maxlen=ReplayMemorySize)

# a=np.full([1,2],0)
# b=np.full([1,2],1)
# c=np.full([1,2],2)

# transition=(a,b,c)

# replayMemory.append(transition)

# a=np.full([1,2],3)
# b=np.full([1,2],4)
# c=np.full([1,2],5)

# transition=(a,b,c)

# replayMemory.append(transition)

# a=np.full([1,2],6)
# b=np.full([1,2],7)
# c=np.full([1,2],8)

# transition=(a,b,c)

# replayMemory.append(transition)


# MinibatchSize=2
# minibatch=random.sample(replayMemory,MinibatchSize)


# # %%
# import torch

# toCheck=np.full([1,5],None)

# x_ls_str=np.full([1,2], None)
# x_ls_str[0,0]='lamp is off'
# x_ls_str[0,1]=torch.tensor([[0]])

# x_cur_str=np.full([1,2], None)
# x_cur_str[0,0]='curtain is closed'
# x_cur_str[0,1]=torch.tensor([[1/2]])

# toCheck[0,0]=x_ls_str
# toCheck[0,1]=x_cur_str



# rule=toCheck[0,0]
# rule1=rule[0,1]
# print(rule1)
# rule=np.concatenate((rule,rule1),axis=1)
# rule[0,2]=torch.tensor([[rule[0,2]]])
# print(rule)

# # %%
# # section 3 
# totalEpochRewards_2_v3_with_53=np.load('section3/totalEpochRewards_2_v3_with_53.npy')

# totalEpochRewards_2_v3_without_53=np.load('section3/totalEpochRewards_2_v3_without_53.npy')
# import matplotlib.pyplot as plt

# fig,ax = plt.subplots()

# plt.plot(totalEpochRewards_2_v3_with_53[0,:4*50],label='with rules',linewidth=4)
# plt.plot(totalEpochRewards_2_v3_without_53[0,:4*50],label='without rules',linewidth=4)





# plt.setp(ax.get_xticklabels(), fontsize=20)
# plt.setp(ax.get_yticklabels(), fontsize=20)



# plt.legend(fontsize=20)

# ax.set_xlabel('epoch',fontsize=20)
# ax.set_ylabel('reward',fontsize=20)
# plt.grid()

# plt.savefig('figs/performance_rl_nn.eps', format='eps')

# plt.show()
# # %%
# # section6
# # import matplotlib.style
# # import matplotlib as mpl
# # from cycler import cycler
# # mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

# x=np.arange(0,100).reshape(1,-1)
# fullEpoch=np.ones((1,100))*288*4

# seenEpochRewards_with_not_seen=np.load('experiment'+'/v3/section6'+f'/seenEpochRewards_100.npy')
# seenStepNums=(seenEpochRewards_with_not_seen/4).astype(np.int32)
# # seenStepNums=(seenEpochRewards_with_not_seen/4)/288

# totalEpochRewards_2_v3_with_not_seen_nn=np.load('experiment'+'/v3/section6'+'/totalEpochRewards_2_v3_with_not_seen_NN_100.npy')

# totalEpochRewards_2_v3_with_not_seen_rules=np.load('experiment'+'/v3/section6'+'/totalEpochRewards_2_v3_with_not_seen_RULES_100.npy')
# import matplotlib.pyplot as plt

# fig,ax = plt.subplots()

# # ax.plot(x[0,:50],fullEpoch[0,:50],label='full',color='r')
# # ax.plot(x[0,:50],totalEpochRewards_2_v3_with_not_seen_nn[0,:50],'o--',color='C0',label='nn')
# # ax.plot(x[0,:50],totalEpochRewards_2_v3_with_not_seen_rules[0,:50],color='g',label='rules')

# ax.plot(x[0,:-1],seenStepNums[0,:-1])
# # ax.bar(x[0,:-1],seenStepNums[0,:-1],label='number of rules not predicted steps')


# # ax.plot(totalEpochRewards_2_v3_with_not_seen_nn[0,:-1]-totalEpochRewards_2_v3_with_not_seen_rules[0,:-1])


# plt.setp(ax.get_xticklabels(), fontsize=20)
# plt.setp(ax.get_yticklabels(), fontsize=20)



# # plt.legend(fontsize=20)

# ax.set_xlabel('epoch',fontsize=20)
# ax.set_ylabel('percentage of \npredictable steps',fontsize=20)
# # plt.grid()

# plt.savefig('figs/performance_rl_nn.eps', format='eps')

# plt.show()



