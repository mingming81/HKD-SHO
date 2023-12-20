
import torch 
import numpy as np
import pandas as pd

from collections import deque

import random

from simulated_light import *
from simulated_temp_3_2_tanh import *
from simulated_co2 import *

from ISTM_TEMP_Service import *


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
    def __init__(self,hidden_layers=600):
        super(LstmServiceModel,self).__init__()
        
        self.num_output_features=110-X_CUR.shape[1]
        
        self.hidden_layers=hidden_layers
        
        self.lstm1=torch.nn.LSTMCell(6,self.hidden_layers)
        self.lstm2=torch.nn.LSTMCell(self.hidden_layers,self.hidden_layers)
        self.lstm3=torch.nn.LSTMCell(self.hidden_layers,self.hidden_layers)
        self.linear=torch.nn.Linear(self.hidden_layers,self.num_output_features)
            
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

class AirService_cell:
    
    def __init__(self,
                 replayMemorySize=1000000,minibatchSize=128,discount=0.1, learningRate=0.9):
        
        self.replayMemorySize=replayMemorySize
        self.minibatchSize=minibatchSize
        self.discount=discount
        self.learningRate=learningRate
        
        self.num_epochs=num_epochs
        
        self.replayMemory=deque(maxlen=self.replayMemorySize)
        
        self.airServiceModel=self.createModel().to(device)
        
        
        self.optimizer=torch.optim.Adam(self.airServiceModel.parameters(),lr=0.1)
        
        
        self.targetAirServiceModel=self.createModel().to(device)
        
        self.targetAirServiceModel.load_state_dict(self.airServiceModel.state_dict())
        
        self.targetAirServiceModel.eval()
        
        # %%
        
    def updateReplayMemory(self,transition):
        self.replayMemory.append(transition)
        
        # %%
        
    def setStates(self,x_us_t,x_ar_t,x_ae_t,x_ap_t,x_win_t,x_cur_t, x_tr_t,x_te_t, x_t_t, x_et_t, MAX_CO2):
        self.x_us_t=x_us_t
        self.x_ar_t=x_ar_t
        self.x_ae_t=x_ae_t
        self.x_ap_t=x_ap_t
        self.x_win_t=x_win_t
        self.x_cur_t=x_cur_t
        self.x_tr_t=x_tr_t
        self.x_te_t=x_te_t
        self.x_t_t=x_t_t
        self.x_et_t=x_et_t
        
        self.MAX_CO2=MAX_CO2
        
        # %%
        
    def getReward(self,x_ar_new_t_2,x_ar_new_t_ori):
        
        if self.x_us_t==torch.tensor([[0]],dtype=torch.float32,device=device): 
            if self.x_ar_t==x_ar_new_t_ori:
                self.reward=200
            else:
                self.reward=-torch.abs(x_ar_new_t_2-x_ar_new_t_ori).item()
                
        if self.x_us_t==torch.tensor([[1]],dtype=torch.float32,device=device):
            if self.x_ar_t<=torch.tensor([[300]],dtype=torch.float32,device=device) and self.x_ar_t>=torch.tensor([[100]],dtype=torch.float32,device=device):
                if self.x_ap_t==torch.tensor([[0]],dtype=torch.float32,device=device):
                    self.reward=200
                
                else:
                    
                    x_ap_new_t_int=torch.tensor([[0]],dtype=torch.float32,device=device)
                    stop=False
                    # for i in range(X_CUR.shape[1]):
                    #     if stop:
                    #         break
                    for j in range(X_WIN.shape[1]):
                        if stop:
                            break
                        for k in range(X_T.shape[1]):
                            if stop:
                                break
                            for k2 in range(X_ET.shape[1]):
                                # x_cur_new_t_int=X_CUR[[[0]],i]
                                x_win_new_t_int=X_WIN[[[0]],j]
                                x_t_new_t_int=X_T[[[0]],k]
                                x_et_new_t_int=X_ET[[[0]],k2]
                                x_ar_new_t_int,_=self.getIndoorAir(self.x_ar_t,x_ap_new_t_int,self.x_cur_t,x_win_new_t_int, x_t_new_t_int,x_et_new_t_int)
                                if x_ar_new_t_int<=torch.tensor([[300]],dtype=torch.float32,device=device) and x_ar_new_t_int>=torch.tensor([[100]],dtype=torch.float32,device=device):
                                    self.reward=-torch.abs(self.x_ar_t-200).item()*2
                                    stop=True
                                    break
                                else:
                                    self.reward=100
            else:
                self.reward=-torch.abs(x_ar_new_t_2-200).item()
                
        if self.x_us_t==torch.tensor([[2]],dtype=torch.float32,device=device):
            if self.x_ar_t<=torch.tensor([[400]],dtype=torch.float32,device=device) and self.x_ar_t>=torch.tensor([[200]],dtype=torch.float32,device=device):
                if self.x_ap_t==torch.tensor([[0]],dtype=torch.float32,device=device):
                    self.reward=200
                
                else:
                    
                    x_ap_new_t_int=torch.tensor([[0]],dtype=torch.float32,device=device)
                    stop=False
                    # for i in range(X_CUR.shape[1]):
                    #     if stop:
                    #         break
                    for j in range(X_WIN.shape[1]):
                        if stop:
                            break
                        for k in range(X_T.shape[1]):
                            if stop:
                                break
                            for k2 in range(X_ET.shape[1]):
                                # x_cur_new_t_int=X_CUR[[[0]],i]
                                x_win_new_t_int=X_WIN[[[0]],j]
                                x_t_new_t_int=X_T[[[0]],k]
                                x_et_new_t_int=X_ET[[[0]],k2]
                                x_ar_new_t_int,_=self.getIndoorAir(self.x_ar_t,x_ap_new_t_int,self.x_cur_t,x_win_new_t_int, x_t_new_t_int,x_et_new_t_int)
                                if x_ar_new_t_int<=torch.tensor([[400]],dtype=torch.float32,device=device) and x_ar_new_t_int>=torch.tensor([[200]],dtype=torch.float32,device=device):
                                    self.reward=-torch.abs(self.x_ar_t-300).item()*2
                                    stop=True
                                    break
                                else:
                                    self.reward=100
            else:
                self.reward=-torch.abs(x_ar_new_t_2-300).item()
        
        if self.x_us_t==torch.tensor([[3]],dtype=torch.float32,device=device): 
            if self.x_ar_t<=torch.tensor([[200]],dtype=torch.float32,device=device) and self.x_ar_t>=torch.tensor([[100]],dtype=torch.float32,device=device):
                if self.x_ap_t==torch.tensor([[0]],dtype=torch.float32,device=device):
                    self.reward=200
                
                else:
                    
                    x_ap_new_t_int=torch.tensor([[0]],dtype=torch.float32,device=device)
                    stop=False
                    # for i in range(X_CUR.shape[1]):
                    #     if stop:
                    #         break
                    for j in range(X_WIN.shape[1]):
                        if stop:
                            break
                        for k in range(X_T.shape[1]):
                            if stop:
                                break
                            for k2 in range(X_ET.shape[1]):
                                # x_cur_new_t_int=X_CUR[[[0]],i]
                                x_win_new_t_int=X_WIN[[[0]],j]
                                x_t_new_t_int=X_T[[[0]],k]
                                x_et_new_t_int=X_ET[[[0]],k2]
                                x_ar_new_t_int,_=self.getIndoorAir(self.x_ar_t,x_ap_new_t_int,self.x_cur_t,x_win_new_t_int, x_t_new_t_int, x_et_new_t_int)
                                if x_ar_new_t_int<=torch.tensor([[200]],dtype=torch.float32,device=device) and x_ar_new_t_int>=torch.tensor([[100]],dtype=torch.float32,device=device):
                                    self.reward=-torch.abs(self.x_ar_t-150).item()*2
                                    stop=True
                                    break
                                else:
                                    self.reward=100
            else:
                self.reward=-torch.abs(x_ar_new_t_2-150).item()
                
        return self.reward
    
    # %%
                              
    def getIndoorAir(self,x_ar_t,x_ap_t,x_cur_t,x_win_t,x_t_t,x_et_t):
        
        pho=1.205
        v=60
        delta_t=x_et_t*60
        delta_t_ap=x_ap_t*x_t_t/60
        delta_t_b=5*60
        
        self.h=2
        
        self.d_l=2
        
        self.d_w=0.2
        
        self.g=9.81
        
        self.lamda=0.019
        
        self.epsilon=1

        exhaled_co2=38000
        
        B_us=torch.tensor([[0,11.004,31.44,7.6635]],dtype=torch.float32,device=device)
        
        n_us=0
        
        
        if self.x_us_t==torch.tensor([[0]],dtype=torch.float32,device=device): 
            n_us=0
        else:
            n_us=1
            
        b_us=B_us[0,int(self.x_us_t.item())]*10e-6
        
        L_ap=torch.tensor([[0,60,170,280,390,500]])
        l_ap=L_ap[0,int(x_ap_t.item())]
        
        openness=torch.min(x_cur_t,x_win_t)
        
        # the air intensity inside the room
        pho_r=1.293*273/(273+self.x_tr_t)
        pho_e=1.293*273/(273+self.x_te_t)


        
        
        # air flow rate
        if self.x_ae_t>=x_ar_t:
            L_t=abs(self.h*openness*self.d_l*torch.sqrt((2*self.g*torch.abs((pho_e-pho_r))*self.h*openness)/(self.lamda*self.d_w*pho_r/self.d_l+self.epsilon*pho_r)))
        else:
            L_t=-abs(self.h*openness*self.d_l*torch.sqrt((2*self.g*torch.abs((pho_e-pho_r))*self.h*openness)/(self.lamda*self.d_w*pho_r/self.d_l+self.epsilon*pho_r)))
            
        x_ar_new_t=self.x_ar_t*(1 - torch.min(torch.tensor([[1]],dtype=torch.float32), (l_ap*delta_t_ap)/v) - torch.min(torch.tensor([[1]],dtype=torch.float32), (L_t*delta_t)/v) - torch.min(torch.tensor([[1]],dtype=torch.float32), (n_us*b_us*delta_t_b)/v))+self.x_ae_t*(torch.min(torch.tensor([[1]],dtype=torch.float32), (L_t*delta_t)/v))+exhaled_co2*torch.min(torch.tensor([[1]],dtype=torch.float32), (n_us*b_us*delta_t_b)/v)

        # print('==================')

        # print(f'self.x_ar_t: {self.x_ar_t}')

        # print('==================')

        # print(f'self.x_ae_t: {self.x_ae_t}')

        # print('==================')

        # print(f'(1. l_ap*delta_t_ap)/v*x_ar_t: {self.x_ar_t*(l_ap*delta_t_ap)/v}')

        # print('==================')

        # print(f'(2. self.x_ar_t*L_t*delta_t)/v: {self.x_ar_t*(L_t*delta_t)/v}')

        # print('==================')

        # print(f'(3. self.x_ar_t*(n_us*b_us*delta_t_b)/v: {self.x_ar_t*(n_us*b_us*delta_t_b)/v}')

        # print('==================')

        # print(f'(4. self.x_ae_t*((L_t*delta_t)/v): {self.x_ae_t*((L_t*delta_t)/v)}')

        # print('==================')

        # print(f'(5. (n_us*b_us*delta_t_b)/v*exhaled_co2: {exhaled_co2*(n_us*b_us*delta_t_b)/v}')

        # print('==================')

        

        # print(f'x_ar_new_t: {x_ar_new_t}')



        # print('==================')
        
        # x_ar_new_t=max(torch.tensor([[0]], dtype=torch.float32),x_ar_new_t.reshape((1,-1)))

        x_ar_new_t_2=copy.deepcopy(x_ar_new_t)
        
        

        x_ar_new_t=torch.min(torch.tensor([[800]],dtype=torch.float32), x_ar_new_t)
        x_ar_new_t=torch.max(torch.tensor([[0]],dtype=torch.float32), x_ar_new_t)
        
        # print(f'2nd x_ar_new_t: {x_ar_new_t}')

        return x_ar_new_t,x_ar_new_t_2
    
    # %%
        
    def createModel(self):
        
       airServiceModel=LstmServiceModel()
        
       return airServiceModel
   
   # %%
    
    def getActions(self,X_t_norm):
        
        Q=self.airServiceModel(X_t_norm.float())
        # Q_cur_t=Q[:,:len(X_CUR[0])].reshape(1,-1)
        Q_win_t=Q[:,:(len(X_WIN[0]))].reshape(1,-1)
        Q_ap_t=Q[:,(len(X_WIN[0])):(len(X_AP[0])+len(X_WIN[0]))].reshape(1,-1)
        Q_t_t=Q[:,(len(X_AP[0])+len(X_WIN[0])):(len(X_AP[0])+len(X_WIN[0])+len(X_T[0]))].reshape(1,-1)
        Q_et_t=Q[:, -len(X_ET[0]):].reshape(1,-1)
        
        return Q_win_t,Q_ap_t,Q_t_t, Q_et_t
        
      # %%
      
      
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
        
        q_list_t=self.airServiceModel(states_list_t.float()).reshape((self.minibatchSize,-1))
        
        # q_cur_list_t=q_list_t[:,:len(X_CUR[0])]
        q_win_list_t=q_list_t[:,:(len(X_WIN[0]))]
        q_ap_list_t=q_list_t[:,(len(X_WIN[0])):(len(X_AP[0])+len(X_WIN[0]))]
        q_t_list_t=q_list_t[:,(len(X_AP[0])+len(X_WIN[0])):(len(X_AP[0])+len(X_WIN[0])+len(X_T[0]))]
        q_et_list_t=q_list_t[:, -len(X_ET[0]):]
        
        states_list_tt_plus_1=[transition[3] for transition in minibatch]
        states_list_t_plus_1=None
        
        for i in range(len(states_list_tt_plus_1)):
            if i==0:
                states_list_t_plus_1=states_list_tt_plus_1[i]
            else:
                states_list_ttt_plus_1=states_list_tt_plus_1[i]
                states_list_t_plus_1=torch.cat((states_list_t_plus_1,states_list_ttt_plus_1),axis=0)
                
        states_list_t_plus_1=states_list_t_plus_1.reshape((self.minibatchSize,-1))
        
        q_list_t_plus_1=self.targetAirServiceModel(states_list_t_plus_1.float())[:,:].detach().reshape((self.minibatchSize,-1))
        # q_cur_list_t_plus_1=q_list_t_plus_1[:,:len(X_CUR[0])]
        q_win_list_t_plus_1=q_list_t_plus_1[:,:(len(X_WIN[0]))]
        q_ap_list_t_plus_1=q_list_t_plus_1[:,(len(X_WIN[0])):(len(X_AP[0])+len(X_WIN[0]))]
        q_t_list_t_plus_1=q_list_t_plus_1[:,(len(X_AP[0])+len(X_WIN[0])):(len(X_AP[0])+len(X_WIN[0])+len(X_T[0]))]
        q_et_list_t_plus_1=q_list_t_plus_1[:, -len(X_ET[0]):]
        
        X=None
        Y=None
        
        for index, (states_t,actions,reward_t,states_t_plus_1) in enumerate(minibatch):
            # if index!=23:
            
            # max_q_cur_t_plus_1=torch.max(q_cur_list_t_plus_1[index],dim=0,keepdim=True)[0]
            # new_q_cur_t_plus_1=reward_t+self.discount*max_q_cur_t_plus_1
            
            max_q_win_t_plus_1=torch.max(q_win_list_t_plus_1[index],dim=0,keepdim=True)[0]
            new_q_win_t_plus_1=reward_t+self.discount*max_q_win_t_plus_1
            
            max_q_ap_t_plus_1=torch.max(q_ap_list_t_plus_1[index],dim=0,keepdim=True)[0]
            new_q_ap_t_plus_1=reward_t+self.discount*max_q_ap_t_plus_1
            
            max_q_t_t_plus_1=torch.max(q_t_list_t_plus_1[index],dim=0,keepdim=True)[0]
            new_q_t_t_plus_1=reward_t+self.discount*max_q_t_t_plus_1
            
            max_q_et_t_plus_1=torch.max(q_et_list_t_plus_1[index],dim=0,keepdim=True)[0]
            new_q_et_t_plus_1=reward_t+self.discount*max_q_et_t_plus_1
            
            
            q_ap_t=q_ap_list_t[index,:]
            q_ap_t=q_ap_t.reshape(1,-1)
            
            q_win_t=q_win_list_t[index,:]
            q_win_t=q_win_t.reshape(1,-1)
            
            # q_cur_t=q_cur_list_t[index,:]
            # q_cur_t=q_cur_t.reshape(1,-1)
            
            q_t_t=q_t_list_t[index,:]
            q_t_t=q_t_t.reshape(1,-1)
            
            q_et_t=q_et_list_t[index,:]
            q_et_t=q_et_t.reshape(1,-1)
            
            action_cur_t,action_win_t,action_ap_t, action_t_t, action_et_t=actions
            
            action_ap_t_item=action_ap_t.item()
            action_ap_t_item=(X_AP==action_ap_t_item).nonzero(as_tuple=True)[1].item()
            
            action_win_t_item=action_win_t.item()
            action_win_t_item=(X_WIN==action_win_t_item).nonzero(as_tuple=True)[1].item()
            
            # action_cur_t_item=action_cur_t.item()
            # action_cur_t_item=(X_CUR==action_cur_t_item).nonzero(as_tuple=True)[1].item()
            
            action_t_t_item=action_t_t.item()
            action_t_t_item=(X_T==action_t_t_item).nonzero(as_tuple=True)[1].item()
            
            action_et_t_item=action_et_t.item()
            action_et_t_item=(X_ET==action_et_t_item).nonzero(as_tuple=True)[1].item()
            
            q_ap_t[0,int(action_ap_t_item)]=(1-self.learningRate)*q_ap_t[0,int(action_ap_t_item)]+self.learningRate*new_q_ap_t_plus_1
            # q_ap_t[0,:int(action_ap_t_item)]=self.learningRate*q_ap_t[0,:int(action_ap_t_item)]-(1-self.learningRate)*reward_t
            # q_ap_t[0,int(action_ap_t_item)+1:]=self.learningRate*q_ap_t[0,int(action_ap_t_item)+1:]-(1-self.learningRate)*reward_t
            
            q_win_t[0,int(action_win_t_item)]=(1-self.learningRate)*q_win_t[0,int(action_win_t_item)]+self.learningRate*new_q_win_t_plus_1
            # q_win_t[0,:int(action_win_t_item)]=self.learningRate*q_win_t[0,:int(action_win_t_item)]-(1-self.learningRate)*reward_t
            # q_win_t[0,int(action_win_t_item)+1:]=self.learningRate*q_win_t[0,int(action_win_t_item)+1:]-(1-self.learningRate)*reward_t
            
            # q_cur_t[0,int(action_cur_t_item)]=(1-self.learningRate)*q_cur_t[0,int(action_cur_t_item)]+self.learningRate*new_q_cur_t_plus_1
            # q_cur_t[0,:int(action_cur_t_item)]=self.learningRate*q_cur_t[0,:int(action_cur_t_item)]-(1-self.learningRate)*reward_t
            # q_cur_t[0,int(action_cur_t_item)+1:]=self.learningRate*q_cur_t[0,int(action_cur_t_item)+1:]-(1-self.learningRate)*reward_t
            
            q_t_t[0,int(action_t_t_item)]=(1-self.learningRate)*q_t_t[0,int(action_t_t_item)]+self.learningRate*new_q_t_t_plus_1
            
            q_et_t[0,int(action_et_t_item)]=(1-self.learningRate)*q_et_t[0,int(action_et_t_item)]+self.learningRate*new_q_et_t_plus_1
            
            if index==0:
                X=copy.deepcopy(states_t)
                Y=torch.cat((q_win_t,q_ap_t, q_t_t, q_et_t),axis=1)
                
            else:
                X=torch.cat((X,states_t),axis=0)
                q_t=torch.cat((q_win_t,q_ap_t, q_t_t, q_et_t),axis=1)
                Y=torch.cat((Y,q_t),axis=0)

        # print(f'Y: {Y.shape}')
                
        data_size=len(X)
        validation_pct=0.2
        
        train_size=math.ceil(data_size*(1-validation_pct))
        
        X_train,Y_train=X[:train_size,:],Y[:train_size,:]
        X_test,Y_test=X[train_size:,:],Y[train_size:,:]

        if epoch>=0:
            self.optimizer=torch.optim.Adam(self.airServiceModel.parameters())
            
        least_val_loss=0
        total_patience=50
        patience=0
        
            
        outputs=self.airServiceModel(X_train.float()).reshape((train_size,-1))
        
        criterion=torch.nn.MSELoss()
        
        # print(f'outputs: {outputs.shape}')
        # print(f'Y_train: {Y_train.shape}')


        loss=criterion(outputs,Y_train)
        
        
        self.optimizer.zero_grad()
        
        loss.backward(retain_graph=True)
        
        self.optimizer.step()
        
        criterion_val=torch.nn.MSELoss()
        
        self.airServiceModel.eval()
        
        total_val_loss=torch.tensor([[0]],dtype=torch.float32,device=device)
        
        with torch.no_grad():
            n_correct=0
            n_samples=0
            
            x_test=X_test
            y_test=Y_test
            
            outputs=self.airServiceModel(x_test.float()).reshape((x_test.shape[0],-1))
            
            val_loss=criterion_val(outputs,y_test)
            
            print(f'val loss: {val_loss.item():.4f}, train loss: {(loss.item()):.4f}')
            
            # if epoch==0:
            #     least_val_loss=total_val_loss
            # else:
            #     if least_val_loss>total_val_loss:
            #         least_val_loss=total_val_loss
            #     else:
            #         patience+=1
            
            # if patience==50:
            #     torch.save(airService.airServiceModel.state_dict(),'data/structure1/airService_v1.pth')
            #     print("end training")
            #     return
        
        self.airServiceModel.train()
        
        return minibatch

# %%

# airService=AirService()
# # airService.airServiceModel.load_state_dict(torch.load('data/lstm/airService_air_lstm_v1_8_sig.pth'))
# # airService.targetAirServiceModel.load_state_dict(torch.load('data/lstm/airService_air_lstm_v1_8_sig.pth'))
# tempService=TempService_cell()

# seenTrainingStates=deque(maxlen=100000)

# totalEpochRewards=np.full([num_epochs,3],0)

# # %%
# for epoch in range(num_epochs):
    
#     # print(f'epoch: {epoch}')
    
#     OneHotEncoder=OneHotEncoderClass()
    
#     x_cur_t=torch.tensor([[0]],dtype=torch.float32,device=device)

#     x_ac_t=torch.tensor([[0]],dtype=torch.float32,device=device)
#     x_ap_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    
#     x_win_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    
   
#     x_tr_t=torch.tensor([[20]],dtype=torch.float32,device=device)
#     x_ar_t=torch.tensor([[200]],dtype=torch.float32,device=device)
#     x_t_t=torch.tensor([[0.1]],dtype=torch.float32,device=device)
#     x_et_t=torch.tensor([[0]],dtype=torch.float32,device=device)


#     X_us_t=user_simulator()

#     X_te_t=temp_simulator()
#     idx_x_te_t=(torch.max(X_te_t,dim=1,keepdim=True)[1]).to(device)
#     MAX_TEMP=X_te_t[[[0]],idx_x_te_t.item()]
#     MAX_TEMP=int(MAX_TEMP)+1
    
#     X_ae_t=co2_simulator()
#     idx_x_ae_t=(torch.max(X_ae_t,dim=1,keepdim=True)[1]).to(device)
#     MAX_CO2=X_ae_t[[[0]],idx_x_ae_t.item()]
#     MAX_CO2=int(MAX_CO2)+1

#     num_corr=0

#     for step in range(0,steps):
        
#         # print(f'epoch: {epoch}')
        
#         # print(f'step: {step}')
        
        
        
#         if step==0:
#             # user states
#             x_us_t=X_us_t[0,step]
#             x_us_t=x_us_t.reshape(1,-1)
   
            
#         else:
#             if step%12==0:
#                 x_us_t=user_simulator()
#             x_us_t=x_us_t.reshape((1,-1))
            
            
#         # outside environment states

#         x_te_t=X_te_t[0,step]
#         x_te_t=x_te_t.reshape((1,-1))


#         x_ae_t=X_ae_t[0,step]
#         x_ae_t=x_ae_t.reshape((1,-1))


#         tempService.setStates(x_us_t, x_tr_t, x_te_t, x_cur_t, x_ac_t, x_win_t, MAX_TEMP, x_t_t,x_et_t)
#         airService.setStates(x_us_t, x_ar_t, x_ae_t, x_ap_t, x_win_t, x_cur_t, x_tr_t, x_te_t, x_t_t, x_et_t, MAX_CO2)


#         x_us_t_norm=OneHotEncoder._one_hot_encoder(X_US,x_us_t)
        
#         # data normalization: light service
#         x_ae_t_norm=x_ae_t/MAX_CO2
#         x_ar_t_norm=x_ar_t/MAX_CO2
#         x_air_t_norm=torch.cat((x_us_t_norm,x_ae_t_norm,x_ar_t_norm),axis=1)

#         sigma=torch.rand(1).item()

#         if sigma>0.1:

#             Q_air_cur_t,Q_air_win_t,Q_air_ap_t,Q_air_t_t, Q_air_et_t=airService.getActions(x_air_t_norm)

#             idx_x_cur_t_new=(torch.max(Q_air_cur_t,dim=1,keepdim=True)[1]).to(device)
#             x_cur_t_new=X_CUR[[[0]],idx_x_cur_t_new.item()]

#             idx_x_win_t_new=(torch.max(Q_air_win_t,dim=1,keepdim=True)[1]).to(device)
#             x_win_t_new=X_WIN[[[0]],idx_x_win_t_new.item()]

#             idx_x_ap_t_new=(torch.max(Q_air_ap_t,dim=1,keepdim=True)[1]).to(device)
#             x_ap_t_new=X_AP[[[0]],idx_x_ap_t_new.item()]
            
            
#             idx_x_t_t_new=(torch.max(Q_air_t_t,dim=1,keepdim=True)[1]).to(device)
#             x_t_t_new=X_T[[[0]],idx_x_t_t_new.item()]
            
#             idx_x_et_t_new=(torch.max(Q_air_et_t,dim=1,keepdim=True)[1]).to(device)
#             x_et_t_new=X_ET[[[0]],idx_x_et_t_new.item()]


#         if sigma<0.1:

#             idx_x_cur_t_new=(torch.randint(0,len(X_CUR[0]),(1,1))).to(device)
#             x_cur_t_new=X_CUR[[[0]],idx_x_cur_t_new]
            
        
#             x_win_t_new=torch.randint(0,len(X_WIN[0]),(1,1)).to(device)

#             x_ap_t_new=torch.randint(0,len(X_AP[0]),(1,1)).to(device)
            
#             idx_x_t_t_new=(torch.randint(0,len(X_T[0]),(1,1))).to(device)
#             x_t_t_new=X_T[[[0]],idx_x_t_t_new]
            
#             idx_x_et_t_new=(torch.randint(0,len(X_ET[0]),(1,1))).to(device)
#             x_et_t_new=X_ET[[[0]],idx_x_et_t_new]


#         x_ac_t_new=copy.deepcopy(x_ac_t)


#         x_tr_t_new=tempService.getIndoorTemp(x_tr_t,x_ac_t_new, x_cur_t_new, x_win_t_new, x_t_t_new, x_et_t_new)
#         x_ar_t_new,x_ar_new_t_2=airService.getIndoorAir(x_ar_t,x_ap_t_new, x_cur_t_new, x_win_t_new, x_t_t_new, x_et_t_new)

#         print(f'x_us_t: {x_us_t}')
#         print(f'x_ae_t: {x_ae_t}')
#         print(f'x_ar_t: {x_ar_t}')
#         print(f'x_ar_t_new: {x_ar_t_new}')
#         print(f'x_ar_new_t_2: {x_ar_new_t_2}')
#         print(f'x_ap_t_new, x_t_t_new: {x_ap_t_new, x_t_t_new}')
#         print(f'x_cur_t_new, x_win_t_new, x_et_t_new: {x_cur_t_new, x_win_t_new, x_et_t_new}')



#         x_us_t_new=copy.deepcopy(x_us_t)

#         x_tr_t_new=copy.deepcopy(x_tr_t_new)
#         x_ar_t_new=copy.deepcopy(x_ar_t_new)

#         x_te_t_new=copy.deepcopy(x_te_t)
#         x_ae_t_new=copy.deepcopy(x_ae_t)

#         x_cur_t_new=copy.deepcopy(x_cur_t_new)
#         x_win_t_new=copy.deepcopy(x_win_t_new)
#         x_ap_t_new=copy.deepcopy(x_ap_t_new)

#         x_us_t_new_norm=OneHotEncoder._one_hot_encoder(X_US, x_us_t_new)

#         x_te_t_new_norm=x_te_t_new/MAX_TEMP

#         # x_temp_t_new_norm=torch.cat((x_us_t_new_norm,x_te_t_new_norm),axis=1)

#         x_ae_t_new_norm=x_ae_t_new/MAX_CO2
#         x_ar_t_new_norm=x_ar_t_new/MAX_CO2
#         x_air_t_new_norm=torch.cat((x_us_t_new_norm,x_ae_t_new_norm,x_ar_t_new),axis=1)
        

#         air_actions=(x_cur_t_new,x_win_t_new,x_ap_t_new, x_t_t_new, x_et_t_new)

#         print(f'x_cur_t_new: {x_cur_t_new}')
#         print(f'x_win_t_new: {x_win_t_new}')
#         print(f'x_ap_t_new: {x_ap_t_new}')
#         print(f'x_ap_t_new: {x_ap_t_new}')



#         airService.setStates(x_us_t_new, x_ar_t_new, x_ae_t_new, x_ap_t_new, x_win_t_new, x_cur_t_new, x_tr_t_new, x_te_t_new, x_t_t_new, x_et_t_new, MAX_CO2)
#         tempService.setStates(x_us_t_new, x_tr_t_new, x_te_t_new, x_cur_t_new, x_ac_t_new, x_win_t_new, MAX_TEMP, x_t_t_new, x_et_t_new)

#         air_reward=airService.getReward(x_ar_new_t_2,x_ar_t)
#         air_reward_t=torch.tensor([[air_reward]],dtype=torch.float32,device=device)

#         totalEpochRewards[epoch,2]+=air_reward


#         if air_reward>0:
#             num_corr+=1

#         print(f'num_corr: {num_corr}')
        

#         # print(f'totalEpochRewards: {totalEpochRewards[epoch,:]}')

#         air_transition=(x_air_t_norm, air_actions, air_reward_t,x_air_t_new_norm)

#         airService.updateReplayMemory(air_transition)

#         _=airService.train(epoch)
        

#         x_ar_t=copy.deepcopy(x_ar_t_new)
#         x_tr_t=copy.deepcopy(x_tr_t_new)
#         x_cur_t=copy.deepcopy(x_cur_t_new)
#         x_win_t=copy.deepcopy(x_win_t_new)
#         x_ap_t=copy.deepcopy(x_ap_t_new)
#         x_t_t=copy.deepcopy(x_t_t_new)
#         x_et_t=copy.deepcopy(x_et_t_new)

# # %%
        
#         # print('yes')
#         print(f'epoch:{epoch}, step:{step}, reward:{air_reward_t}, totalEpochRewards:{totalEpochRewards[epoch,2]}')

#         print(f'=================================================================================')
        
#         # %%

#     if epoch%5==0:
#         airService.targetAirServiceModel.load_state_dict(airService.airServiceModel.state_dict())


#     if epoch%200==0 and epoch!=0:    
        
#         torch.save(airService.airServiceModel.state_dict(),f'data/lstm/airService_air_lstm_v1_{1+int(epoch/200)}_sig_range_8_v1.pth')
#         np.save(f'data/lstm/totalEpochRewards_air_lstm_v1_{1+int(epoch/200)}_sig_range_8_v1.npy',totalEpochRewards)


# torch.save(airService.airServiceModel.state_dict(),'data/lstm/airService_air_lstm_v1_sig_range_8_v1.pth')
# np.save(f'data/lstm/totalEpochRewards_air_lstm_v1_sig_range_8_v1.npy',totalEpochRewards)
    
