
import torch 
import numpy as np
import pandas as pd

from collections import deque

import random

from simulated_light import *
from simulated_temp_3_2_tanh import *
# from simulated_co2 import *


import copy

import math


# %%

random.seed(0)

torch.autograd.set_detect_anomaly(True)

device=torch.device('cpu')
# device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# %%

batch_size=120
num_epochs=1000
steps=288

# %%
X_US=torch.tensor([[0,1,2,3]],dtype=torch.float32,device=device)

X_LS=torch.tensor([[0,1,2,3,4]],dtype=torch.float32,device=device)

X_CUR=torch.tensor([[0,1/2,1]],dtype=torch.float32, device=device)

X_WIN=torch.tensor([[0,1]],dtype=torch.float32,device=device)

X_AC=torch.tensor([[0,1,-1]],dtype=torch.float32,device=device)

X_T=torch.tensor([[i/10 for i in range(0,50)]],dtype=torch.float32,device=device)
X_T=X_T[[0],1:]

X_ET=torch.tensor([[i/10 for i in range(0,50)]],dtype=torch.float32,device=device)

# X_AC=torch.cat((torch.tensor([[0]],dtype=torch.float32,device=device),X_AC),axis=1)

X_AP=torch.tensor([[0,1,2,3,4,5]],dtype=torch.float32,device=device)

# %%

class LstmServiceModel(torch.nn.Module):
    def __init__(self,hidden_layers=200):
        super(LstmServiceModel,self).__init__()
        
        self.num_output_features=107-X_CUR.shape[1]-X_WIN.shape[1]-X_ET.shape[1]
        
        self.hidden_layers=hidden_layers
        
        self.lstm1=torch.nn.LSTMCell(6,self.hidden_layers)
        self.lstm2=torch.nn.LSTMCell(self.hidden_layers,self.hidden_layers)
        self.lstm3=torch.nn.LSTMCell(self.hidden_layers,self.hidden_layers)
        self.linear=torch.nn.Linear(self.hidden_layers,self.num_output_features)
        
        # h_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        # c_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        # h_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        # c_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        # h_t3 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        # c_t3 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
            
    def forward(self,y):
       
        
        if len(y.shape)!=3:
            y=torch.from_numpy(np.expand_dims(y, axis=0))
            
    
        n_samples =y.size(0)
        
        h_t, c_t = self.lstm1(y[0])
        h_t2, c_t2 = self.lstm2(h_t)
        h_t3, c_t3 = self.lstm2(h_t2)
        
        output = self.linear(h_t3)
       

        return output
    
# %%

def user_simulator():
    X_us_t=torch.randint(0,4,(1,1)).to(device)
    return X_us_t



class OneHotEncoderClass:
    def __init__(self):
        pass
    def _one_hot_encoder(self,X,x):
        
        zeros=torch.zeros(X.shape,dtype=torch.float32,device=device)
        # print(f'zeros shape: {zeros.shape}')
        pos=torch.where(X==x)[1].item()
        zeros[0,pos]=1
        one_hot_encod=zeros
        
        return one_hot_encod
    
# %%

class TempService_cell:
    
    def __init__(self,
                 
                 replayMemorySize=1000000,minibatchSize=128,
                 discount=0.1, learningRate=0.9):
        # self.theta_ac=theta_ac
        # self.cp=cp
        # self.pho=pho
        
        # self.v=v
        
        # self.h=h
        
        # self.d_l=d_l
        
        # self.d_w=d_w
        
        # self.g=g
        
        # self.lamda=lamda
        
        # self.epsilon=epsilon
        
        self.replayMemorySize=replayMemorySize
        self.minibatchSize=minibatchSize
        self.discount=discount
        self.learningRate=learningRate
        
        self.num_epochs=num_epochs
        
        self.replayMemory=deque(maxlen=self.replayMemorySize)
        
        self.tempServiceModel=self.createModel().to(device)
        
        
        self.optimizer=torch.optim.Adam(self.tempServiceModel.parameters(),lr=0.1)
        
        self.targetTempServiceModel=self.createModel().to(device)
        self.targetTempServiceModel.load_state_dict(self.tempServiceModel.state_dict())

        
        self.targetTempServiceModel.eval()
        
    def updateReplayMemory(self,transition):
        self.replayMemory.append(transition)
        
    def setStates(self,x_us_t,x_tr_t,x_te_t,x_cur_t,x_ac_t,x_win_t,MAX_TEMP,x_t_t,x_et_t):
        self.x_us_t=x_us_t
        self.x_tr_t=x_tr_t
        self.x_te_t=x_te_t
        self.x_cur_t=x_cur_t
        self.x_ac_t=x_ac_t
        self.x_win_t=x_win_t
        self.MAX_TEMP=MAX_TEMP
        self.x_t_t=x_t_t
        self.x_et_t=x_et_t
        
    def getReward(self,x_tr_new_t_2,x_tr_new_t_ori):
        
        if self.x_us_t==torch.tensor([[0]],dtype=torch.float32,device=device): 
            if self.x_tr_t==x_tr_new_t_ori:
                self.reward=8
            else:
                self.reward=-torch.abs(x_tr_new_t_2-x_tr_new_t_ori).item()
                
        if self.x_us_t==torch.tensor([[1]],dtype=torch.float32,device=device):
            if self.x_tr_t>=torch.tensor([[23]],dtype=torch.float32,device=device) and self.x_tr_t<=torch.tensor([[25]],dtype=torch.float32,device=device):
                # if self.x_ac_t==torch.tensor([[0]],dtype=torch.float32,device=device):
                self.reward=8
                
                # else:
                    
                #     x_ac_new_t_int=torch.tensor([[0]],dtype=torch.float32,device=device)
                #     stop=False
                #     for i in range(X_CUR.shape[1]):
                #         if stop:
                #             break
                #         for j in range(X_WIN.shape[1]):
                #             if stop:
                #                 break
                #             for k in range(X_T.shape[1]):
                #                 if stop:
                #                     break
                #                 for k2 in range(X_ET.shape[1]):
                #                     x_cur_new_t_int=X_CUR[[[0]],i]
                #                     x_win_new_t_int=X_WIN[[[0]],j]
                #                     x_t_new_t_int=X_T[[[0]],k]
                #                     x_et_new_t_int=X_ET[[[0]],k2]
                #                     x_tr_new_t_int,_=self.getIndoorTemp(self.x_tr_t,x_ac_new_t_int,x_cur_new_t_int,x_win_new_t_int,x_t_new_t_int, x_et_new_t_int)
                #                     if x_tr_new_t_int>=torch.tensor([[23]],dtype=torch.float32,device=device) and x_tr_new_t_int<=torch.tensor([[25]],dtype=torch.float32,device=device):
                #                         self.reward=-torch.abs(self.x_tr_t-24).item()*2
                #                         stop=True
                #                         break
                #                     else:
                #                         self.reward=4
            else:
                self.reward=-torch.abs(x_tr_new_t_2-24).item()
                
        if self.x_us_t==torch.tensor([[2]],dtype=torch.float32,device=device):
            if self.x_tr_t>=torch.tensor([[20]],dtype=torch.float32,device=device) and self.x_tr_t<=torch.tensor([[22]],dtype=torch.float32,device=device):
                # if self.x_ac_t==torch.tensor([[0]],dtype=torch.float32,device=device):
                self.reward=8
                # else:
                #     x_ac_new_t_int=torch.tensor([[0]],dtype=torch.float32,device=device)
                #     stop=False
                #     for i in range(X_CUR.shape[1]):
                #         if stop:
                #             break
                #         for j in range(X_WIN.shape[1]):
                #             if stop:
                #                 break
                #             for k in range(X_T.shape[1]):
                #                 if stop:
                #                     break
                                
                #                 for k2 in range(X_ET.shape[1]):
                #                     x_cur_new_t_int=X_CUR[[[0]],i]
                #                     x_win_new_t_int=X_WIN[[[0]],j]
                #                     x_t_new_t_int=X_T[[[0]],k]
                #                     x_et_new_t_int=X_ET[[[0]],k2]

                #                     x_tr_new_t_int,_=self.getIndoorTemp(self.x_tr_t,x_ac_new_t_int,x_cur_new_t_int,x_win_new_t_int,x_t_new_t_int, x_et_new_t_int)
                #                     if x_tr_new_t_int>=torch.tensor([[20]],dtype=torch.float32,device=device) and x_tr_new_t_int<=torch.tensor([[22]],dtype=torch.float32,device=device):
                #                         self.reward=-torch.abs(self.x_tr_t-21).item()*2
                #                         stop=True
                #                         break
                #                     else:
                #                         self.reward=4
                                    
            else:
                self.reward=-torch.abs(x_tr_new_t_2-21).item()
                
        if self.x_us_t==torch.tensor([[3]],dtype=torch.float32,device=device): 
            if self.x_tr_t>=torch.tensor([[17]],dtype=torch.float32,device=device) and self.x_tr_t<=torch.tensor([[19]],dtype=torch.float32,device=device):
                # if self.x_ac_t==torch.tensor([[0]],dtype=torch.float32,device=device):
                self.reward=8
                # else:
                #     x_ac_new_t_int=torch.tensor([[0]],dtype=torch.float32,device=device)
                #     stop=False
                #     for i in range(X_CUR.shape[1]):
                #         if stop:
                #             break
                #         for j in range(X_WIN.shape[1]):
                #             if stop:
                #                 break
                #             for k in range(X_T.shape[1]):
                #                 if stop:
                #                     break
                #                 for k2 in range(X_ET.shape[1]):
                #                     x_cur_new_t_int=X_CUR[[[0]],i]
                #                     x_win_new_t_int=X_WIN[[[0]],j]
                #                     x_t_new_t_int=X_T[[[0]],k]
                #                     x_et_new_t_int=X_ET[[[0]],k2]
                #                     x_tr_new_t_int,_=self.getIndoorTemp(self.x_tr_t,x_ac_new_t_int,x_cur_new_t_int,x_win_new_t_int,x_t_new_t_int,x_et_new_t_int)
                #                     if x_tr_new_t_int>=torch.tensor([[17]],dtype=torch.float32,device=device) and x_tr_new_t_int<=torch.tensor([[19]],dtype=torch.float32,device=device):
                #                         self.reward=-torch.abs(self.x_tr_t-18).item()*2
                #                         stop=True
                #                         break
                #                     else:
                #                         self.reward=4
                                    
            else:
                self.reward=-torch.abs(x_tr_new_t_2-18).item()
                
        return self.reward
                            
    
    
    def getIndoorTemp(self,x_tr_t,x_ac_t,x_cur_t,x_win_t,x_t_t,x_et_t):

        theta_ac=20.0*735.0
        cp=1.005
        pho=1.205
        v=60.0
        h=2.0
        d_l=2.0
        d_w=0.2
        g=9.81
        lamda=0.019
        epsilon=1.0
        
        delta_t=x_et_t*60 
        delta_t_ac=x_ac_t*x_t_t/60 # 3 minutes heating
        e_ap=theta_ac*delta_t_ac

        # print(f'e_ap: {e_ap}')
        # print(f'e_ap temp: {e_ap.item()/(cp*pho*v)}')
        # # print(f'========')
        
        
        openness=torch.min(x_cur_t,x_win_t)

        # print(f'x_win_t: {x_win_t}')
        # print(f'x_cur_t: {x_cur_t}')
        # print(f'openness: {openness}')
        
        # print(f'========')
        
        # the air intensity inside the room
        pho_r=1.293*273/(273+self.x_tr_t)
        pho_e=1.293*273/(273+self.x_te_t)
        
        # print(f'pho_r: {pho_r}')
        # print(f'pho_e: {pho_e}')
        
        # print(f'========')
        
        
        # air flow rate
        if self.x_te_t>=self.x_tr_t:
            L_t=h*openness*d_l*torch.sqrt((2*g*torch.abs((pho_e-pho_r))*h*openness)/(lamda*d_w*pho_r/d_l+epsilon*pho_r))
        else:
            L_t=-h*openness*d_l*torch.sqrt((2*g*torch.abs((pho_e-pho_r))*h*openness)/(lamda*d_w*pho_r/d_l+epsilon*pho_r))
        
        e_env=L_t*delta_t*pho*cp

        # print(f'e_env: {e_env}')
        # print(f'e_env temp: {e_env.item()/(cp*pho*v)}')
        # print(f'========')
        
        e=e_ap+e_env
        
        diff_t=e/(cp*pho*v)

        # print(f'diff_t: {diff_t}')
        
        x_tr_new_t=x_tr_t+diff_t

        x_tr_new_t_2=copy.deepcopy(x_tr_new_t)

        
        # print(f'========')

        x_tr_new_t=torch.min(torch.tensor([[26]], dtype=torch.float32), x_tr_new_t)      
        x_tr_new_t=torch.max(torch.tensor([[12]], dtype=torch.float32), x_tr_new_t)        
        
        # x_tr_new_t=x_tr_new_t.reshape((1,-1))
        return x_tr_new_t,x_tr_new_t_2
        
        
        
    def createModel(self):
        # hidden : 26, 58
        tempServiceModel=LstmServiceModel()
        
        return tempServiceModel
    
    def getActions(self,X_t_norm):
        
        
        
        print(f'X_t_norm: {X_t_norm}')
        
        Q=self.tempServiceModel(X_t_norm.float())
        
        
        # Q_cur_t=Q[:,:len(X_CUR[0])].reshape(1,-1)
        # Q_ac_t=Q[:,len(X_CUR[0]):(len(X_CUR[0])+len(X_AC[0]))].reshape(1,-1)
        # Q_win_t=Q[:,len(X_CUR[0])+len(X_AC[0]):(len(X_CUR[0])+len(X_AC[0])+len(X_WIN[0]))].reshape(1,-1)
        # Q_t_t=Q[:,(len(X_CUR[0])+len(X_AC[0]))+len(X_WIN[0]):(len(X_CUR[0])+len(X_AC[0])+len(X_WIN[0])+len(X_T[0]))].reshape(1,-1)
        
        # Q_et_t=Q[:, -len(X_ET[0]):].reshape(1,-1)

        Q_ac_t=Q[:,:X_AC.shape[1]].reshape(1,-1)
        Q_t_t=Q[:,X_AC.shape[1]:].reshape((1,-1))

        return Q_ac_t, Q_t_t
        
        
        # idx_x_ac_new_t=(torch.max(Q_ac_t,dim=1,keepdim=True)[1]).to(device)
        # x_ac_new_t=X_AC[[[0]],idx_x_ac_new_t.item()]
        
        # idx_x_win_new_t=(torch.max(Q_win_t,dim=1,keepdim=True)[1]).to(device)
        # x_win_new_t=X_AC[[[0]],idx_x_win_new_t.item()]
        
        # idx_x_cur_new_t=(torch.max(Q_cur_t,dim=1,keepdim=True)[1]).to(device)
        # x_cur_new_t=X_CUR[[[0]],idx_x_cur_new_t.item()]
        
        # return x_ac_new_t,x_win_new_t,x_cur_new_t,idx_x_ac_new_t,idx_x_win_new_t,idx_x_cur_new_t
        
    def train(self,epoch):
        if len(self.replayMemory) < self.minibatchSize:
            return 
        
        minibatch=random.sample(self.replayMemory,self.minibatchSize)
        
        states_list_tt=[transition[0] for transition in minibatch]
        
        states_list_t=None
        
        for i in range(len(states_list_tt)):
            if i==0:
                states_list_t=states_list_tt[i]
            else:
                states_list_ttt=states_list_tt[i]
                states_list_t=torch.cat((states_list_t,states_list_ttt),axis=0)
                
        states_list_t=states_list_t.reshape((self.minibatchSize,-1))
        
        q_list_t=self.tempServiceModel(states_list_t.float()).reshape((self.minibatchSize,-1))

        q_ac_list_t=q_list_t[:,:X_AC.shape[1]]
        q_t_list_t=q_list_t[:,X_AC.shape[1]:]
        
        # q_cur_list_t=q_list_t[:,:len(X_CUR[0])]
        # q_ac_list_t=q_list_t[:,len(X_CUR[0]):(len(X_CUR[0])+len(X_AC[0]))]
        # q_win_list_t=q_list_t[:,(len(X_CUR[0])+len(X_AC[0])):(len(X_CUR[0])+len(X_AC[0]))+len(X_WIN[0])]
        # q_t_list_t=q_list_t[:,(len(X_CUR[0])+len(X_AC[0]))+len(X_WIN[0]):(len(X_CUR[0])+len(X_AC[0])+len(X_WIN[0])+len(X_T[0]))]
        # q_et_list_t=q_list_t[:, -len(X_ET[0]):]
        
        
        states_list_tt_plus_1=[transition[3] for transition in minibatch]
        
        states_list_t_plus_1=None
        
        for i in range(len(states_list_tt_plus_1)):
            if i==0:
                states_list_t_plus_1=states_list_tt_plus_1[i]
            else:
                states_list_ttt_plus_1=states_list_tt_plus_1[i]
                states_list_t_plus_1=torch.cat((states_list_t_plus_1,states_list_ttt_plus_1),axis=0)
                
        states_list_t_plus_1=states_list_t_plus_1.reshape((self.minibatchSize,-1))
        
        q_list_t_plus_1=self.targetTempServiceModel(states_list_t_plus_1.float())[:,:].detach().reshape((self.minibatchSize,-1))

        q_ac_list_t_plus_1=q_list_t_plus_1[:,:X_AC.shape[1]]
        q_t_list_t_plus_1=q_list_t_plus_1[:,X_AC.shape[1]:]


        # q_cur_list_t_plus_1=q_list_t_plus_1[:,:len(X_CUR[0])]
        # q_ac_list_t_plus_1=q_list_t_plus_1[:,len(X_CUR[0]):(len(X_CUR[0])+len(X_AC[0]))]
        # q_win_list_t_plus_1=q_list_t_plus_1[:,(len(X_CUR[0])+len(X_AC[0])):(len(X_CUR[0])+len(X_AC[0]))+len(X_WIN[0])]
        # q_t_list_t_plus_1=q_list_t_plus_1[:,(len(X_CUR[0])+len(X_AC[0]))+len(X_WIN[0]):(len(X_CUR[0])+len(X_AC[0])+len(X_WIN[0])+len(X_T[0]))]
        # q_et_list_t_plus_1=q_list_t_plus_1[:, -len(X_ET[0]):]
        

        X=None
        Y=None
        
        for index, (states_t,actions,reward_t,states_t_plus_1) in enumerate(minibatch):
            # max_q_cur_t_plus_1=torch.max(q_cur_list_t_plus_1[index],dim=0,keepdim=True)[0]
            # new_q_cur_t_plus_1=reward_t+self.discount*max_q_cur_t_plus_1
            
            max_q_ac_t_plus_1=torch.max(q_ac_list_t_plus_1[index],dim=0,keepdim=True)[0]
            new_q_ac_t_plus_1=reward_t+self.discount*max_q_ac_t_plus_1
            
            # max_q_win_t_plus_1=torch.max(q_win_list_t_plus_1[index],dim=0,keepdim=True)[0]
            # new_q_win_t_plus_1=reward_t+self.discount*max_q_win_t_plus_1
            
            max_q_t_t_plus_1=torch.max(q_t_list_t_plus_1[index],dim=0,keepdim=True)[0]
            new_q_t_t_plus_1=reward_t+self.discount*max_q_t_t_plus_1
            
            # max_q_et_t_plus_1=torch.max(q_et_list_t_plus_1[index],dim=0,keepdim=True)[0]
            # new_q_et_t_plus_1=reward_t+self.discount*max_q_et_t_plus_1
            
            
            
            q_ac_t=q_ac_list_t[index,:]
            q_ac_t=q_ac_t.reshape(1,-1)
            
            # q_win_t=q_win_list_t[index,:]
            # q_win_t=q_win_t.reshape(1,-1)
            
            # q_cur_t=q_cur_list_t[index,:]
            # q_cur_t=q_cur_t.reshape(1,-1)
            
            q_t_t=q_t_list_t[index,:]
            q_t_t=q_t_t.reshape(1,-1)
            
            # q_et_t=q_et_list_t[index,:]
            # q_et_t=q_et_t.reshape(1,-1)
            
            # action_cur_t,action_ac_t, action_win_t,action_t_t, action_et_t=actions

            action_ac_t, action_t_t=actions
            
            action_ac_t_item=action_ac_t.item()
            action_ac_t_item=(X_AC==action_ac_t_item).nonzero(as_tuple=True)[1].item()
            
            # action_win_t_item=action_win_t.item()
            # action_win_t_item=(X_WIN==action_win_t_item).nonzero(as_tuple=True)[1].item()
            
            # action_cur_t_item=action_cur_t.item()
            # action_cur_t_item=(X_CUR==action_cur_t_item).nonzero(as_tuple=True)[1].item()
            
            action_t_t_item=action_t_t.item()
            action_t_t_item=(X_T==action_t_t_item).nonzero(as_tuple=True)[1].item()
            
            # action_et_t_item=action_et_t.item()
            # action_et_t_item=(X_ET==action_et_t_item).nonzero(as_tuple=True)[1].item()
            
            q_ac_t[0,int(action_ac_t_item)]=(1-self.learningRate)*q_ac_t[0,int(action_ac_t_item)]+self.learningRate*new_q_ac_t_plus_1
            # q_ac_t[0,:int(action_ac_t_item)]=self.learningRate*q_ac_t[0,:int(action_ac_t_item)]-(1-self.learningRate)*reward_t
            # q_ac_t[0,int(action_ac_t_item)+1:]=self.learningRate*q_ac_t[0,int(action_ac_t_item)+1:]-(1-self.learningRate)*reward_t
            
            # q_win_t[0,int(action_win_t_item)]=(1-self.learningRate)*q_win_t[0,int(action_win_t_item)]+self.learningRate*new_q_win_t_plus_1
            # q_win_t[0,:int(action_win_t_item)]=self.learningRate*q_win_t[0,:int(action_win_t_item)]-(1-self.learningRate)*reward_t
            # q_win_t[0,int(action_win_t_item)+1:]=self.learningRate*q_win_t[0,int(action_win_t_item)+1:]-(1-self.learningRate)*reward_t
            
            # q_cur_t[0,int(action_cur_t_item)]=(1-self.learningRate)*q_cur_t[0,int(action_cur_t_item)]+self.learningRate*new_q_cur_t_plus_1
            # q_cur_t[0,:int(action_cur_t_item)]=self.learningRate*q_cur_t[0,:int(action_cur_t_item)]-(1-self.learningRate)*reward_t
            # q_cur_t[0,int(action_cur_t_item)+1:]=self.learningRate*q_cur_t[0,int(action_cur_t_item)+1:]-(1-self.learningRate)*reward_t
            
            q_t_t[0,int(action_t_t_item)]=(1-self.learningRate)*q_t_t[0,int(action_t_t_item)]+self.learningRate*new_q_t_t_plus_1
            
            # q_et_t[0,int(action_et_t_item)]=(1-self.learningRate)*q_et_t[0,int(action_et_t_item)]+self.learningRate*new_q_et_t_plus_1
            
            if index==0:
                X=copy.deepcopy(states_t)
                Y=torch.cat((q_ac_t,q_t_t),axis=1)
                
            else:
                X=torch.cat((X,states_t),axis=0)
                q_t=torch.cat((q_ac_t,q_t_t),axis=1)
                Y=torch.cat((Y,q_t),axis=0)
                
        data_size=len(X)
        validation_pct=0.2
        
        train_size=math.ceil(data_size*(1-validation_pct))
        
        X_train,Y_train=X[:train_size,:],Y[:train_size,:]
        X_test,Y_test=X[train_size:,:],Y[train_size:,:]

        if epoch>=0:
            self.optimizer=torch.optim.Adam(self.tempServiceModel.parameters())
            
        least_val_loss=0
        total_patience=50
        patience=0
        
            
        outputs=self.tempServiceModel(X_train.float()).reshape((train_size,-1))
        
        criterion=torch.nn.MSELoss()
        
        loss=criterion(outputs,Y_train)
        
        
        self.optimizer.zero_grad()
        
        loss.backward(retain_graph=True)
        
        self.optimizer.step()
        
        criterion_val=torch.nn.MSELoss()
        
        self.tempServiceModel.eval()
        
        total_val_loss=torch.tensor([[0]],dtype=torch.float32,device=device)
        
        with torch.no_grad():
            n_correct=0
            n_samples=0
            
            x_test=X_test
            y_test=Y_test
            
            outputs=self.tempServiceModel(x_test.float()).reshape((x_test.shape[0],-1))
            
            val_loss=criterion_val(outputs,y_test)
            
            print(f'val loss: {val_loss.item():.4f}, train loss: {(loss.item()):.4f}')
            
         
        self.tempServiceModel.train()
        
        return minibatch


