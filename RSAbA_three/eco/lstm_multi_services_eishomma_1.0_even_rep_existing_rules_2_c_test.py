
import torch 
import numpy as np
import pandas as pd

from collections import deque

import random

from simulated_light import *
from simulated_temp_3_2_tanh import *
from simulated_co2 import *

from LSTM_LIGHT_SERVICE_8_CELL import *
from LSTM_AIR_SERVICE_RANGE_8_v1_CELL_EISHOMMA_1_0_c import *
from LSTM_TEMP_SERVICE_88_CELL_EISHOMMA_1_0_c import *


import copy

import math

import itertools
from scipy.spatial import distance

from py4j.java_gateway import JavaGateway, GatewayParameters

# %%

random.seed(0)

torch.manual_seed(0)

np.random.seed(0)

torch.autograd.set_detect_anomaly(True)

device=torch.device('cpu')
# device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# %%

batch_size=120
num_epochs=100
steps=288

# %%

X_US=torch.tensor([[0,1,2,3]],dtype=torch.float32,device=device)

X_US_STR=np.array([['absent','working','playing','sleeping']])

X_LS=torch.tensor([[0,1,2,3,4]],dtype=torch.float32,device=device)
X_LS_STR=np.array([['off','level_1','level_2','level_3','level_4']])

X_CUR=torch.tensor([[0,1/2,1]],dtype=torch.float32, device=device)
X_CUR_STR=np.array([['close','half_open','open']])

X_WIN=torch.tensor([[0,1]],dtype=torch.float32,device=device)
X_WIN_STR=np.array([['close','open']])

X_AC=torch.tensor([[0,1,-1]],dtype=torch.float32,device=device)
X_AC_STR=np.array([['off','heating','cooling']])

X_T=torch.tensor([[i/10 for i in range(0,50)]],dtype=torch.float32,device=device)
X_T=X_T[[0],1:]

# X_AT=torch.tensor([[i/10 for i in range(0,50)]],dtype=torch.float32,device=device)
# X_AT=X_T[[0],1:]

X_TET=torch.tensor([[i/10 for i in range(0,50)]],dtype=torch.float32,device=device)

X_AET=torch.tensor([[i/10 for i in range(0,50)]],dtype=torch.float32,device=device)

X_ET=copy.deepcopy(X_AET)

X_AP=torch.tensor([[0,1,2,3,4,5]],dtype=torch.float32,device=device)
X_AP_STR=np.array([['off','level_1','level_2','level_3','level_4','level_5']])

# %%

def user_simulator():
    X_us_t=torch.randint(0,4,(1,1)).to(device)
    return X_us_t

# %%

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

def decodeUserState(x_us_t):
    idx=int(x_us_t.item())
    x_us_str=X_US_STR[0,idx].item()
    return x_us_str

def encodeCurState(x_cur_t_str):
    idx_tensor=(X_CUR_STR == x_cur_t_str).nonzero()[1][0]
    idx=idx_tensor
    x_cur_t=X_CUR[0,idx].reshape(1,-1)
    return x_cur_t

def encodeLsState(x_ls_t_str):
    idx_tensor=(X_LS_STR == x_ls_t_str).nonzero()[1][0]
    idx=idx_tensor
    x_ls_t=X_LS[0,idx].reshape(1,-1)
    return x_ls_t

def encodeWinState(x_win_t_str):
    idx_tensor=(X_WIN_STR == x_win_t_str).nonzero()[1][0]
    idx=idx_tensor
    x_win_t=X_WIN[0,idx].reshape(1,-1)
    return x_win_t

def encodeAcState(x_ac_t_str):
    idx_tensor=(X_AC_STR == x_ac_t_str).nonzero()[1][0]
    idx=idx_tensor
    x_ac_t=X_AC[0,idx].reshape(1,-1)
    return x_ac_t

def encodeApState(x_ap_t_str):
    idx_tensor=(X_AP_STR == x_ap_t_str).nonzero()[1][0]
    idx=idx_tensor
    x_ap_t=X_AP[0,idx].reshape(1,-1)
    return x_ap_t

    
# %%

lightService=LightService_cell()
# lightService=LightService_cell()
lightService.lightServiceModel.load_state_dict(torch.load('data/lstm/lightService_lstm_v2_5_eishomma_1.0_even_rep_existing_rules_2_c.pth'))
lightService.targetLightServiceModel.load_state_dict(torch.load('data/lstm/lightService_lstm_v2_5_eishomma_1.0_even_rep_existing_rules_2_c.pth'))

tempService=TempService_cell()
tempService.tempServiceModel.load_state_dict(torch.load('data/lstm/tempService_lstm_v2_5_eishomma_1.0_even_rep_existing_rules_2_c.pth'))
tempService.targetTempServiceModel.load_state_dict(torch.load('data/lstm/tempService_lstm_v2_5_eishomma_1.0_even_rep_existing_rules_2_c.pth'))

airService=AirService_cell()
airService.airServiceModel.load_state_dict(torch.load('data/lstm/airService_lstm_v2_5_eishomma_1.0_even_rep_existing_rules_2_c.pth'))
airService.targetAirServiceModel.load_state_dict(torch.load('data/lstm/airService_lstm_v2_5_eishomma_1.0_even_rep_existing_rules_2_c.pth'))



X_US_2=torch.from_numpy(np.load(f'data/lstm/X_US_lstm_v1_eishomma_even_100.npy',allow_pickle=True))
X_LE_2=torch.from_numpy(np.load(f'data/lstm/X_LE_lstm_v1_eishomma_even_100.npy',allow_pickle=True))
X_TE_2=torch.from_numpy(np.load(f'data/lstm/X_TE_lstm_v1_eishomma_even_100.npy',allow_pickle=True))
X_AE_2=torch.from_numpy(np.load(f'data/lstm/X_AE_lstm_v1_eishomma_even_100.npy',allow_pickle=True))




# cur_calculator=CalculationPriorities_cell(num_inputs=X_US.shape[1]+1+1+1,num_outputs=3)
# cur_calculator.calPrioModel.load_state_dict(torch.load('data/structure1/cur_calculator_v4.pth'))
# cur_calculator.targetCalPrioModel.load_state_dict(torch.load('data/structure1/cur_calculator_v4.pth'))

# win_calculator=CalculationPriorities_cell(num_inputs=X_US.shape[1]+1+1,num_outputs=2)
# win_calculator.calPrioModel.load_state_dict(torch.load('data/structure1/win_calculator_v4.pth'))
# win_calculator.targetCalPrioModel.load_state_dict(torch.load('data/structure1/win_calculator_v4.pth'))

seenTrainingStates=deque(maxlen=100000)

totalEpochRewards=np.full([num_epochs,3],0)

# posRelayMemory=np.full((steps*num_epochs*10,30),999,dtype='float32')

posRelayMemory=np.load(f'data/lstm/posReplayMemory_lstm_v2_5_eishomma_1.0_even_rep_existing_rules_2_c.npy',allow_pickle=True)

rules=np.full([6,num_epochs],0)

lenRules=np.full([steps,num_epochs],0)

totalAcc=np.full([num_epochs,4],0)


MIN_LIGHT=20
MIN_TEMP=2
MIN_AIR=15

active_rule=None

# %%

gateway = JavaGateway(gateway_parameters=GatewayParameters(port=15351))                   
random = gateway.jvm.java.util.Random()   

hybrid_system = gateway.entry_point               
hybrid_system.loading("./smartHome.owl")

# %%
for epoch in range(num_epochs):
    
    # print(f'epoch: {epoch}')
    
    OneHotEncoder=OneHotEncoderClass()
    
    x_cur_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    x_ls_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    
    x_ac_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    x_win_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    x_tt_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    x_et_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    x_at_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    
    x_ap_t=torch.tensor([[0]],dtype=torch.float32,device=device)
    
    x_lr_t=torch.tensor([[0]],dtype=torch.float32, device=device)
            
    x_tr_t=torch.tensor([[20]],dtype=torch.float32,device=device)
            
    x_ar_t=torch.tensor([[200]],dtype=torch.float32,device=device)
    
    # %%
    
    X_le_t=X_LE_2[[epoch],:]
    idx_x_le_t=(torch.max(X_le_t,dim=1,keepdim=True)[1]).to(device)
    MAX_LE=X_le_t[[[0]],idx_x_le_t.item()]
    MAX_LE=int(MAX_LE)+1
    
    X_te_t=X_TE_2[[epoch],:]
    idx_x_te_t=(torch.max(X_te_t,dim=1,keepdim=True)[1]).to(device)
    MAX_TEMP=X_te_t[[[0]],idx_x_te_t.item()]
    MAX_TEMP=int(MAX_TEMP)+1
    
    
    X_ae_t=X_AE_2[[epoch],:]
    idx_x_ae_t=(torch.max(X_ae_t,dim=1,keepdim=True)[1]).to(device)
    MAX_CO2=X_ae_t[[[0]],idx_x_ae_t.item()]
    MAX_CO2=int(MAX_CO2)+1
    
    
# %%  
    
    num_corr_light=0
    num_corr_temp=0
    num_corr_air=0 
    
    for step in range(0,steps):
        
        # print(f'epoch: {epoch}')
        
        # print(f'step: {step}')

        ruldid=False

        existingRuldid=False

        notRule=False
        
        
        
        
        x_us_t=X_US_2[epoch,step]
        x_us_t=x_us_t.reshape((1,-1))


        value = hybrid_system.getState("nacer");
        print(f'user state before change: {value}')
        
        newValue=decodeUserState(x_us_t)
        
        hybrid_system.runChangeState("nacer",newValue)
        hybrid_system.loading("./smartHome.owl")
        value = hybrid_system.getState("nacer");
        print(f'user state after change: {value}')
            
            
        # outside environment states
        x_le_t=X_le_t[0,step]
        x_le_t=x_le_t.reshape((1,-1))
        
        x_te_t=X_te_t[0,step]
        x_te_t=x_te_t.reshape((1,-1))
        
        x_ae_t=X_ae_t[0,step]
        x_ae_t=x_ae_t.reshape((1,-1))
        
        # service setting environments

        lightService.setStates(x_us_t, x_lr_t, x_le_t, x_ls_t, x_cur_t, MAX_LE)
        
        tempService.setStates(x_us_t, x_tr_t, x_te_t, x_cur_t, x_ac_t, x_win_t, MAX_TEMP,x_tt_t, x_et_t)
        
        airService.setStates(x_us_t, x_ar_t, x_ae_t, x_ap_t, x_win_t, x_cur_t, x_tr_t, x_te_t, x_at_t, x_et_t, MAX_CO2)
        
        
        
        
        # data normalization
        x_us_t_norm=OneHotEncoder._one_hot_encoder(X_US,x_us_t)
        
        # data normalization: light service
        x_le_t_norm=x_le_t/MAX_LE
        x_light_t_norm=torch.cat((x_us_t_norm,x_le_t_norm),axis=1)
        
        # data normalization: temperature service
        x_te_t_norm=x_te_t/MAX_TEMP
        x_tr_t_norm=x_tr_t/MAX_TEMP
        x_temp_t_norm=torch.cat((x_us_t_norm,x_te_t_norm,x_tr_t_norm),axis=1)
        
        # data normalization: air quality service
        x_ae_t_norm=x_ae_t/MAX_CO2
        x_ar_t_norm=x_ar_t/MAX_CO2
        x_air_t_norm=torch.cat((x_us_t_norm,x_ae_t_norm,x_ar_t_norm),axis=1)

        x_t_pos_norm=torch.cat((x_us_t, x_le_t, x_te_t,x_ae_t, x_tr_t, x_ar_t),axis=1)
        
        
        # x_cur_p_t_norm=torch.cat((x_us_t_norm,x_le_t_norm,x_te_t_norm,x_ae_t_norm),axis=1)
        
        
        
        # x_win_p_t_norm=torch.cat((x_us_t_norm,x_te_t_norm,x_ae_t_norm),axis=1)
        
        
        
        sigma=torch.rand(1).item()


        if posRelayMemory[0,14]!=999:

            k_=(np.where(posRelayMemory[:,14]!=999)[0]).shape[0]
            print(f'posReplayMemory size: {k_}')

            lenRules[step,epoch]=k_

            idx_notNone=np.where(posRelayMemory[:,14]!=999)[0]

            actionCandicateReplayMemory=posRelayMemory[(idx_notNone.shape[0]-k_):idx_notNone.shape[0],:]

            actionCandicateReplayMemory=np.flip(actionCandicateReplayMemory,axis=0)

            actionCandicateReplayMemory=np.array(sorted(actionCandicateReplayMemory,key=lambda x: sum(x[14:17]), reverse=True)).reshape(-1,30)

            stateCandidiate=actionCandicateReplayMemory[:,:6]

            # print('=======')

            # print(f'x_t_pos_norm: {x_t_pos_norm}')

            # print('=======')

            # print(f'stateCandidiate: {stateCandidiate}')

            # print('=======')

            actionCandidate=actionCandicateReplayMemory[:,6:14]

            rewardCandidate=actionCandicateReplayMemory[:,14:17]

            maxCandidate=actionCandicateReplayMemory[:,-3:]

            idx_keep=np.where(stateCandidiate[:,:1]==x_t_pos_norm.numpy()[:,:1])[0]

            if idx_keep.shape[0]!=0:

                stateCandidiate=stateCandidiate[idx_keep,:]
                transition_stateCandidate=actionCandicateReplayMemory[idx_keep,:]
                actionCandidate=actionCandidate[idx_keep,:]
                maxCandidate=maxCandidate[idx_keep,:]

                stateCandidiate_light=stateCandidiate[:,[0,1]]
                stateCandidiate_temp=stateCandidiate[:,[0,2,4]]
                stateCandidiate_air=stateCandidiate[:,[0,3,5]]

                x_t_pos_norm_light=x_t_pos_norm[:,[0,1]]
                x_t_pos_norm_temp=x_t_pos_norm[:,[0,2,4]]
                x_t_pos_norm_air=x_t_pos_norm[:,[0,3,5]]


                stateCandidiate2_light=copy.deepcopy(stateCandidiate_light)
                stateCandidiate2_light[:,0]/=4
                stateCandidiate2_light[:,1]/=maxCandidate[:,0]
                # stateCandidiate2[:,2]/=maxCandidate[:,0]

                x_t_pos_norm2_light=copy.deepcopy(x_t_pos_norm_light)
                x_t_pos_norm2_light[0,0]/=4
                x_t_pos_norm2_light[0,1]/=MAX_LE
                # x_t_pos_norm2[0,2]/=MAX_LE


                stateCandidiate2_temp=copy.deepcopy(stateCandidiate_temp)

                stateCandidiate2_temp[:,0]/=4
                stateCandidiate2_temp[:,1]/=maxCandidate[:,1]
                stateCandidiate2_temp[:,2]/=maxCandidate[:,1]

                x_t_pos_norm2_temp=copy.deepcopy(x_t_pos_norm_temp)

                x_t_pos_norm2_temp[0,0]/=4
                x_t_pos_norm2_temp[0,1]/=MAX_TEMP
                x_t_pos_norm2_temp[0,2]/=MAX_TEMP

                stateCandidiate2_air=copy.deepcopy(stateCandidiate_air)
                stateCandidiate2_air[:,0]/=4
                stateCandidiate2_air[:,1]/=maxCandidate[:,2]
                stateCandidiate2_air[:,2]/=maxCandidate[:,2]

                x_t_pos_norm2_air=copy.deepcopy(x_t_pos_norm_air)
                x_t_pos_norm2_air[0,0]/=4
                x_t_pos_norm2_air[0,1]/=MAX_CO2
                x_t_pos_norm2_air[0,2]/=MAX_CO2

                # print(f'=======')
                # print(f'stateCandidiate: {stateCandidiate}')
                # print(f'=======')
                # print(f'x_t_pos_norm: {x_t_pos_norm}')
                # print(f'=======')



                corr2_light=np.array([distance.cosine(np.array([i],dtype='float32'),x_t_pos_norm2_light) for i in stateCandidiate2_light]).reshape(-1)
                corr2_euclidean_light=np.array([distance.euclidean(i,x_t_pos_norm_light) for i in stateCandidiate_light]).reshape(-1)

                # print(f'corr2_light: {corr2_light}')
                # print(f'corr2_euclidean_light: {corr2_euclidean_light}')
                # print(f'=======')

                corr_argsort_light=np.argsort(corr2_light).reshape(-1)
                idx_corr_light=np.array([])

                for i in corr_argsort_light:
                    if corr2_light[i]<0.0002 and corr2_euclidean_light[i]<MIN_LIGHT:
                        idx_corr_light=np.concatenate((idx_corr_light,np.array([i])))



                corr2_temp=np.array([distance.euclidean(np.array([i],dtype='float32'),x_t_pos_norm2_temp) for i in stateCandidiate2_temp]).reshape(-1)
                corr2_euclidean_temp=np.array([distance.euclidean(i,x_t_pos_norm_temp) for i in stateCandidiate_temp]).reshape(-1)

                # print(f'corr2_temp: {corr2_temp}')
                # print(f'corr2_euclidean_temp: {corr2_euclidean_temp}')
                # print(f'=======')

                corr_argsort_temp=np.argsort(corr2_temp).reshape(-1)
                idx_corr_temp=np.array([])

                for i in corr_argsort_temp:
                    if corr2_temp[i]<0.1 and corr2_euclidean_temp[i]<MIN_TEMP:
                        idx_corr_temp=np.concatenate((idx_corr_temp,np.array([i])))


                corr2_air=np.array([distance.cosine(np.array([i],dtype='float32'),x_t_pos_norm2_air) for i in stateCandidiate2_air]).reshape(-1)
                corr2_euclidean_air=np.array([distance.euclidean(i,x_t_pos_norm_air) for i in stateCandidiate_air]).reshape(-1)

                # print(f'corr2_air: {corr2_air}')
                # print(f'corr2_euclidean_air: {corr2_euclidean_air}')
                # print(f'=======')

                corr_argsort_air=np.argsort(corr2_air).reshape(-1)
                idx_corr_air=np.array([])

                for i in corr_argsort_air:
                    if corr2_air[i]<0.0002 and corr2_euclidean_air[i]<MIN_AIR:
                        idx_corr_air=np.concatenate((idx_corr_air,np.array([i])))



                for i in range(stateCandidiate_light.shape[0]):
                    if transition_stateCandidate[i,17]<=x_t_pos_norm[0,1] and transition_stateCandidate[i,18]>=x_t_pos_norm[0,1]:
                        if i not in idx_corr_light:
                            idx_corr_light=np.concatenate((idx_corr_light,np.array([i])))



                for i in range(stateCandidiate_temp.shape[0]):
                    if transition_stateCandidate[i,19]<=x_t_pos_norm[0,2] and transition_stateCandidate[i,20]>=x_t_pos_norm[0,2]:
                        if transition_stateCandidate[i,23]<=x_t_pos_norm[0,4] and transition_stateCandidate[i,24]>=x_t_pos_norm[0,4]:
                            if i not in idx_corr_temp:
                                idx_corr_temp=np.concatenate((idx_corr_temp,np.array([i])))


                for i in range(stateCandidiate_air.shape[0]):
                    if transition_stateCandidate[i,21]<=x_t_pos_norm[0,3] and transition_stateCandidate[i,22]>=x_t_pos_norm[0,3]:
                        if transition_stateCandidate[i,25]<=x_t_pos_norm[0,5] and transition_stateCandidate[i,26]>=x_t_pos_norm[0,5]:
                            if i not in idx_corr_air:
                                idx_corr_air=np.concatenate((idx_corr_air,np.array([i])))

                idx_corr_light=idx_corr_light.astype('int')
                # print(f'idx_corr_light: {idx_corr_light}')
                # print(f'=======')

                idx_corr_temp=idx_corr_temp.astype('int')
                # print(f'idx_corr_temp: {idx_corr_temp}')
                # print(f'=======')

                idx_corr_air=idx_corr_air.astype('int')
                # print(f'idx_corr_air: {idx_corr_air}')
                # print(f'=======')

                # print(f'before action: {x_ls_t_new.item(),x_cur_t_new.item(),x_win_t_new.item(),x_ac_t_new.item(),x_tt_t_new.item(),x_at_t_new.item(),x_et_t_new.item()}')

        
                idx_light_temp=idx_corr_light[np.in1d(idx_corr_light, idx_corr_temp)]
                idx_light_temp_air=idx_light_temp[np.in1d(idx_light_temp,idx_corr_air)]

                if idx_corr_light.shape[0]!=0:

                    # ruldid=True

                    # rules[0,epoch]+=1

                    actionCandidate_light=actionCandidate[idx_corr_light,:][:,[0,1]]
                    note_ls=np.zeros((1,X_LS.shape[1]))
                    note_cur=np.zeros((1,X_CUR.shape[1]))

                    if idx_light_temp_air.shape[0]!=0:

                        ruldid=True

                        # idx_corr_light=np.concatenate((idx_light_temp_air, np.delete(idx_corr_light,[i in idx_light_temp_air for i in idx_corr_light])))

                        idx_corr_light=idx_light_temp_air

                        active_rule=transition_stateCandidate[idx_light_temp_air,:].reshape(idx_light_temp_air.shape[0],-1)

                        actionCandidate_light=actionCandidate[idx_corr_light,:][:,[0,1]]

                        # idx_ls=np.where(X_LS==x_ls_t_new)[1]
                        # note_ls[0,idx_ls[0]]+=1/2**(actionCandidate_light.shape[0])

                        # idx_cur=np.where(X_CUR==x_cur_t_new)[1]
                        # note_cur[0,idx_cur[0]]+=1/2**(actionCandidate_light.shape[0])

                        for i in range(actionCandidate_light.shape[0]):
                            idx_ls=np.where(X_LS==actionCandidate_light[i,0])[1]
                            note_ls[0,idx_ls[0]]+=1/2**(1+i)

                            idx_cur=np.where(X_CUR==actionCandidate_light[i,1])[1]
                            note_cur[0,idx_cur[0]]+=1/2**(1+i)

                    # else:

                    #     idx_ls=np.where(X_LS==x_ls_t_new)[1]
                    #     note_ls[0,idx_ls[0]]+=1/2

                    #     idx_cur=np.where(X_CUR==x_cur_t_new)[1]
                    #     note_cur[0,idx_cur[0]]+=1/2

                    #     for i in range(actionCandidate_light.shape[0]):
                    #         idx_ls=np.where(X_LS==actionCandidate_light[i,0])[1]
                    #         note_ls[0,idx_ls[0]]+=1/2**(2+i)

                    #         idx_cur=np.where(X_CUR==actionCandidate_light[i,1])[1]
                    #         note_cur[0,idx_cur[0]]+=1/2**(2+i)

                    argmax_ls=np.argmax(note_ls,axis=1)
                    x_ls_t_new=X_LS[[0],argmax_ls[0]]

                    argmax_cur=np.argmax(note_cur,axis=1)
                    x_cur_t_new=X_CUR[[0],argmax_cur[0]]


                if idx_corr_temp.shape[0]!=0:

                    # ruldid=True


                    # rules[1,epoch]+=1

                    actionCandidate_temp=actionCandidate[idx_corr_temp,:][:,[3,5]]

                    note_ac=np.zeros((1,X_AC.shape[1]))
                    note_tt=np.zeros((1,X_T.shape[1]))

                    # idx_temp_light=idx_corr_temp[np.in1d(idx_corr_temp, idx_corr_light)]
                    # idx_temp_light_air=idx_temp_light[np.in1d(idx_temp_light,idx_corr_air)]

                    if idx_light_temp_air.shape[0]!=0:

                        # idx_corr_temp=np.concatenate((idx_light_temp_air, np.delete(idx_corr_temp,[i in idx_light_temp_air for i in idx_corr_temp])))

                        idx_corr_temp=idx_light_temp_air

                        actionCandidate_temp=actionCandidate[idx_corr_temp,:][:,[3,5]]

                        # idx_ac=np.where(X_AC==x_ac_t_new)[1]
                        # note_ac[0,idx_ac[0]]+=1/2**(actionCandidate_temp.shape[0])

                        # idx_tt=np.where(X_T==x_tt_t_new)[1]
                        # note_tt[0,idx_tt[0]]+=1/2**(actionCandidate_temp.shape[0])

                        for i in range(actionCandidate_temp.shape[0]):

                            idx_ac=np.where(X_AC==actionCandidate_temp[i,0])[1]
                            note_ac[0,idx_ac[0]]+=1/2**(1+i)

                            idx_tt=np.where(X_T==actionCandidate_temp[i,1])[1]
                            note_tt[0,idx_tt[0]]+=1/2**(1+i)

                    # else:

                    #     idx_ac=np.where(X_AC==x_ac_t_new)[1]
                    #     note_ac[0,idx_ac[0]]+=1/2

                    #     idx_tt=np.where(X_T==x_tt_t_new)[1]
                    #     note_tt[0,idx_tt[0]]+=1/2

                    #     for i in range(actionCandidate_temp.shape[0]):

                    #         idx_ac=np.where(X_AC==actionCandidate_temp[i,0])[1]
                    #         note_ac[0,idx_ac[0]]+=1/2**(2+i)

                    #         idx_tt=np.where(X_T==actionCandidate_temp[i,1])[1]
                    #         note_tt[0,idx_tt[0]]+=1/2**(2+i)

                    argmax_ac=np.argmax(note_ac,axis=1)
                    x_ac_t_new=X_AC[[0],argmax_ac[0]]

                    argmax_tt=np.argmax(note_tt,axis=1)
                    x_tt_t_new=X_T[[0],argmax_tt[0]]

                if idx_corr_air.shape[0]!=0:

                    # ruldid=True

                    # rules[2,epoch]+=1
                    actionCandidate_air=actionCandidate[idx_corr_air,:][:,[2,4,6,7]]

                    note_win=np.zeros((1,X_WIN.shape[1]))
                    note_ap=np.zeros((1,X_AP.shape[1]))
                    note_at=np.zeros((1,X_T.shape[1]))
                    note_et=np.zeros((1,X_ET.shape[1]))

                    # idx_air_light=idx_corr_air[np.in1d(idx_corr_air, idx_corr_light)]
                    # idx_air_light_temp=idx_air_light[np.in1d(idx_air_light,idx_corr_temp)]

                    if idx_light_temp_air.shape[0]!=0:

                        # idx_corr_air=np.concatenate((idx_light_temp_air, np.delete(idx_corr_air,[i in idx_light_temp_air for i in idx_corr_air])))

                        idx_corr_air=idx_light_temp_air

                        actionCandidate_air=actionCandidate[idx_corr_air,:][:,[2,4,6,7]]

                        # idx_win=np.where(X_WIN==x_win_t_new)[1]
                        # note_win[0,idx_win[0]]+=1/2**(actionCandidate_air.shape[0])

                        # idx_ap=np.where(X_AP==x_ap_t_new)[1]
                        # note_ap[0,idx_ap[0]]+=1/2**(actionCandidate_air.shape[0])

                        # idx_at=np.where(X_T==x_at_t_new)[1]
                        # note_at[0,idx_at[0]]+=1/2**(actionCandidate_air.shape[0])

                        # idx_et=np.where(X_ET==x_et_t_new)[1]
                        # note_et[0,idx_et[0]]+=1/2**(actionCandidate_air.shape[0])

                        for i in range(actionCandidate_air.shape[0]):

                            idx_win=np.where(X_WIN==actionCandidate_air[i,0])[1]
                            note_win[0,idx_win[0]]+=1/2**(1+i)

                            idx_ap=np.where(X_AP==actionCandidate_air[i,1])[1]
                            note_ap[0,idx_ap[0]]+=1/2**(1+i)

                            idx_at=np.where(X_T==actionCandidate_air[i,2])[1]
                            note_at[0,idx_at[0]]+=1/2**(1+i)

                            idx_et=np.where(X_ET==actionCandidate_air[i,3])[1]
                            note_et[0,idx_et[0]]+=1/2**(1+i)

                    # else:

                    #     idx_win=np.where(X_WIN==x_win_t_new)[1]
                    #     note_win[0,idx_win[0]]+=1/2

                    #     idx_ap=np.where(X_AP==x_ap_t_new)[1]
                    #     note_ap[0,idx_ap[0]]+=1/2

                    #     idx_at=np.where(X_T==x_at_t_new)[1]
                    #     note_at[0,idx_at[0]]+=1/2

                    #     idx_et=np.where(X_ET==x_et_t_new)[1]
                    #     note_et[0,idx_et[0]]+=1/2

                    #     for i in range(actionCandidate_air.shape[0]):

                    #         idx_win=np.where(X_WIN==actionCandidate_air[i,0])[1]
                    #         note_win[0,idx_win[0]]+=1/2**(2+i)

                    #         idx_ap=np.where(X_AP==actionCandidate_air[i,1])[1]
                    #         note_ap[0,idx_ap[0]]+=1/2**(2+i)

                    #         idx_at=np.where(X_T==actionCandidate_air[i,2])[1]
                    #         note_at[0,idx_at[0]]+=1/2**(2+i)

                    #         idx_et=np.where(X_ET==actionCandidate_air[i,3])[1]
                    #         note_et[0,idx_et[0]]+=1/2**(2+i)

                    argmax_win=np.argmax(note_win,axis=1)
                    x_win_t_new=X_WIN[[0],argmax_win[0]]

                    argmax_ap=np.argmax(note_ap,axis=1)
                    x_ap_t_new=X_AP[[0],argmax_ap[0]]

                    argmax_at=np.argmax(note_at,axis=1)
                    x_at_t_new=X_T[[0],argmax_at[0]]

                    argmax_et=np.argmax(note_et,axis=1)
                    x_et_t_new=X_ET[[0],argmax_et[0]]


        num_existRule_decision=0

        # hybrid_system.loading("./smartHome.owl")
        x_cur_t_new_str=hybrid_system.getState("curtain");
        # print(f'existing rule x_cur_t_new_str: {x_cur_t_new_str}')
        if x_cur_t_new_str!=None:
            x_cur_t_new=encodeCurState(x_cur_t_new_str)
            existingRuldid=True
            num_existRule_decision+=1
            # print(f'existing rule x_cur_t_new: {x_cur_t_new}')
            
        x_ls_t_new_str=hybrid_system.getState("lamp");
        # print(f'existing rule x_ls_t_new_str: {x_ls_t_new_str}')
        if x_ls_t_new_str!=None:
            x_ls_t_new=encodeLsState(x_ls_t_new_str)
            existingRuldid=True
            num_existRule_decision+=1
            # print(f'existing rule x_ls_t_new: {x_ls_t_new}')
            
        x_win_t_new_str=hybrid_system.getState("window");
        # print(f'existing rule x_win_t_new_str: {x_win_t_new_str}')
        if x_win_t_new_str!=None:
            x_win_t_new=encodeWinState(x_win_t_new_str)
            existingRuldid=True
            num_existRule_decision+=1
            # print(f'existing rule x_win_t_new: {x_win_t_new}')
            
        x_ac_t_new_str=hybrid_system.getState("airConditioner");
        # print(f'existing rule x_ac_t_new_str: {x_ac_t_new_str}')
        if x_ac_t_new_str!=None:
            x_ac_t_new=encodeAcState(x_ac_t_new_str)
            existingRuldid=True
            num_existRule_decision+=1
            # print(f'existing rule x_ac_t_new: {x_ac_t_new}')
            
        x_ap_t_new_str=hybrid_system.getState("airPurifier");
        # print(f'existing rule x_ap_t_new_str: {x_ap_t_new_str}')
        if x_ap_t_new_str!=None:
            x_ap_t_new=encodeApState(x_ap_t_new_str)
            existingRuldid=True
            num_existRule_decision+=1
            # print(f'existing rule x_ap_t_new: {x_ap_t_new}')

        if ruldid==True:
            notRule=False
        else:
            notRule=True

        if existingRuldid==5:
            notRule=False

        if notRule==True:
            
      
            if sigma>0.1:
                # curtain priorities
                # cur_priorities=cur_calculator.getPriorities(x_cur_p_t_norm)
                # cur_light_p_t=cur_priorities[0,[0]]
                cur_light_p_t_norm=torch.tensor([[1]], dtype=torch.float32, device=device)
                # cur_temp_p_t=cur_priorities[0,[1]]
                cur_temp_p_t_norm=torch.tensor([[1]], dtype=torch.float32, device=device)
                # cur_air_p_t=cur_priorities[0,[2]]
                cur_air_p_t_norm=torch.tensor([[1]], dtype=torch.float32, device=device)
                
                # window priorities
                # win_priorities=win_calculator.getPriorities(x_win_p_t_norm)
                # win_temp_p_t=win_priorities[0,[0]]
                win_temp_p_t_norm=torch.tensor([[1]], dtype=torch.float32, device=device)
                # win_air_p_t=win_priorities[0,[1]]
                win_air_p_t_norm=torch.tensor([[1]], dtype=torch.float32, device=device)

                t_temp_p_t_norm=torch.tensor([[1]],dtype=torch.float32,device=device)
                t_air_p_t_norm=torch.tensor([[1]],dtype=torch.float32,device=device)
                


                # action qualities: services
                Q_light_cur_t,Q_light_ls_t=lightService.getActions(x_light_t_norm)
                Q_temp_ac_t,Q_temp_tt_t=tempService.getActions(x_temp_t_norm)
                Q_air_win_t,Q_air_ap_t,Q_air_at_t, Q_air_et_t=airService.getActions(x_air_t_norm)
                
                
                # action qualities:curtain
                # Q_light_cur_t_norm=Q_light_cur_t/(torch.abs(Q_light_cur_t).sum())
                # # idx_x_cur_t_new_light=(torch.max(Q_light_cur_t_norm,dim=1,keepdim=True)[1]).to(device)
                
                # Q_temp_cur_t_norm=Q_temp_cur_t/(torch.abs(Q_temp_cur_t).sum())
                # # idx_x_cur_t_new_temp=(torch.max(Q_temp_cur_t_norm,dim=1,keepdim=True)[1]).to(device)
                
                # Q_air_cur_t_norm=Q_air_cur_t/(torch.abs(Q_air_cur_t).sum())
                # idx_x_cur_t_new_air=(torch.max(Q_air_cur_t_norm,dim=1,keepdim=True)[1]).to(device)
                
                # Q_cur_t_norm=cur_light_p_t_norm*Q_light_cur_t_norm+cur_temp_p_t_norm*Q_temp_cur_t_norm+cur_air_p_t_norm*Q_air_cur_t_norm
                
                idx_x_cur_t_new=(torch.max(Q_light_cur_t,dim=1,keepdim=True)[1]).to(device)
                x_cur_t_new=X_CUR[[[0]],idx_x_cur_t_new.item()]
                
                
                # action qualities:lamp
                x_ls_t_new=torch.max(Q_light_ls_t,dim=1,keepdim=True)[1].to(device)
                
                # action qualities:air conditioner
                idx_x_ac_t_new=(torch.max(Q_temp_ac_t,dim=1,keepdim=True)[1]).to(device)
                x_ac_t_new=X_AC[[[0]],idx_x_ac_t_new.item()]

                
                
                # action qualities:window
                # Q_temp_win_t_norm=Q_temp_win_t/(torch.abs(Q_temp_win_t).sum())
                # Q_air_win_t_norm=Q_air_win_t/(torch.abs(Q_air_win_t).sum())
                # Q_win_t_norm=win_temp_p_t_norm*Q_temp_win_t_norm+win_air_p_t_norm*Q_air_win_t_norm
                
                idx_x_win_t_new=(torch.max(Q_air_win_t,dim=1,keepdim=True)[1]).to(device)
                x_win_t_new=X_WIN[[[0]],idx_x_win_t_new.item()]





                # Q_temp_et_t_norm=Q_temp_et_t/(torch.abs(Q_temp_et_t).sum())
                # Q_air_et_t_norm=Q_air_et_t/(torch.abs(Q_air_et_t).sum())
                # Q_et_t_norm=t_temp_p_t_norm*Q_temp_et_t_norm+t_air_p_t_norm*Q_air_et_t_norm

                idx_x_et_t_new=(torch.max(Q_air_et_t,dim=1,keepdim=True)[1]).to(device)
                x_et_t_new=X_ET[[0],idx_x_et_t_new.item()]



                idx_x_tt_t_new=(torch.max(Q_temp_tt_t,dim=1,keepdim=True)[1]).to(device)
                x_tt_t_new=X_T[[[0]],idx_x_tt_t_new.item()]


                idx_x_at_t_new=(torch.max(Q_air_at_t,dim=1,keepdim=True)[1]).to(device)
                x_at_t_new=X_T[[[0]],idx_x_at_t_new.item()]

                
                # action qualities: air purifier
                idx_x_ap_t_new=(torch.max(Q_air_ap_t,dim=1,keepdim=True)[1]).to(device)
                x_ap_t_new=X_AP[[[0]],idx_x_ap_t_new.item()]
                

            if sigma<0.1:
                # action selection:curtain
                idx_x_cur_t_new=(torch.randint(0,len(X_CUR[0]),(1,1))).to(device)
                x_cur_t_new=X_CUR[[[0]],idx_x_cur_t_new]
                
                # action selection:lamp
                x_ls_t_new=torch.randint(0,len(X_LS[0]),(1,1)).to(device)
                
                # action selection:air conditioner
                idx_x_ac_t_new=(torch.randint(0,len(X_AC[0]),(1,1))).to(device)
                x_ac_t_new=X_AC[[[0]],idx_x_ac_t_new]

                idx_x_tt_t_new=(torch.randint(0,len(X_T[0]),(1,1))).to(device)
                x_tt_t_new=X_T[[[0]],idx_x_tt_t_new]
                
                # action selection: window
                x_win_t_new=torch.randint(0,len(X_WIN[0]),(1,1)).to(device)


                idx_x_at_t_new=(torch.randint(0,len(X_T[0]),(1,1))).to(device)
                x_at_t_new=X_T[[[0]],idx_x_at_t_new]
                

                idx_x_et_t_new=(torch.randint(0,len(X_ET[0]),(1,1))).to(device)
                x_et_t_new=X_ET[[[0]],idx_x_et_t_new]

               
                # action selection: air purifier
                x_ap_t_new=torch.randint(0,len(X_AP[0]),(1,1)).to(device)



        # update the indoor states
        x_lr_t_new,x_lr_new_t_2=lightService.getIndoorLight(x_ls_t_new,x_cur_t_new)
        x_tr_t_new,x_tr_new_t_2=tempService.getIndoorTemp(x_tr_t,x_ac_t_new, x_cur_t_new, x_win_t_new, x_tt_t_new, x_et_t_new)
        x_ar_t_new,x_ar_new_t_2=airService.getIndoorAir(x_ar_t,x_ap_t_new, x_cur_t_new, x_win_t_new, x_at_t_new, x_et_t_new)

        # print(f'x_us_t: {x_us_t}')

        # print(f'x_le_t: {x_le_t}')
        # print(f'x_lr_t: {x_lr_t}')
        # print(f'x_lr_t_new: {x_lr_t_new}')
        # print(f'x_lr_new_t_2: {x_lr_new_t_2}')
        

        # print(f'x_te_t: {x_te_t}')
        # print(f'x_tr_t: {x_tr_t}')
        # print(f'x_tr_t_new: {x_tr_t_new}')
        # print(f'x_tr_new_t_2: {x_tr_new_t_2}')

        # print(f'x_ae_t: {x_ae_t}')
        # print(f'x_ar_t: {x_ar_t}')
        # print(f'x_ar_t_new: {x_ar_t_new}')
        # print(f'x_ar_new_t_2: {x_ar_new_t_2}')

        
        # update the current state values
        x_us_t_new=copy.deepcopy(x_us_t)
        
        # x_lr_t_new=x_lr_t_new
        x_tr_t_new=copy.deepcopy(x_tr_t_new)
        x_ar_t_new=x_ar_t_new
        
        x_le_t_new=x_le_t
        x_te_t_new=copy.deepcopy(x_te_t)
        x_ae_t_new=x_ae_t
        
        x_cur_t_new=copy.deepcopy(x_cur_t_new)
        # x_ls_t_new=copy.deepcopy(x_ls_t_new)
        x_ac_t_new=copy.deepcopy(x_ac_t_new)
        x_tt_t_new=copy.deepcopy(x_tt_t_new)
        x_at_t_new=copy.deepcopy(x_at_t_new)
        x_et_t_new=copy.deepcopy(x_et_t_new)
        x_win_t_new=copy.deepcopy(x_win_t_new)
        x_ap_t_new=copy.deepcopy(x_ap_t_new)
        
        # new data normalization
        x_us_t_new_norm=OneHotEncoder._one_hot_encoder(X_US, x_us_t_new)
        
        # data normalization: light service
        x_le_t_new_norm=x_le_t_new/MAX_LE
        x_light_t_new_norm=torch.cat((x_us_t_new_norm,x_le_t_new_norm),axis=1)
        
        # data normalization: temperature service
        x_te_t_new_norm=x_te_t_new/MAX_TEMP
        x_tr_t_new_norm=x_tr_t_new/MAX_TEMP
        x_temp_t_new_norm=torch.cat((x_us_t_new_norm,x_te_t_new_norm,x_tr_t_new_norm),axis=1)
        
        # data normalization: air quality service
        x_ae_t_new_norm=x_ae_t_new/MAX_CO2
        x_ar_t_new_norm=x_ar_t_new/MAX_CO2
        x_air_t_new_norm=torch.cat((x_us_t_new_norm,x_ae_t_new_norm,x_ar_t_new_norm),axis=1)
        
        
        # x_cur_p_t_new_norm=torch.cat((x_us_t_new_norm,x_le_t_new_norm,x_te_t_new_norm,x_ae_t_new_norm),axis=1)
        
        
        # x_win_p_t_new_norm=torch.cat((x_us_t_new_norm,x_te_t_new_norm,x_ae_t_new_norm),axis=1)
        
        light_actions=(x_cur_t_new,x_ls_t_new)

        temp_actions=(x_ac_t_new,x_tt_t_new)

        air_actions=(x_cur_t_new,x_win_t_new,x_ap_t_new, x_at_t_new, x_et_t_new)

        allActions=(x_ls_t_new,x_cur_t_new,x_win_t_new,x_ac_t_new,x_ap_t_new,x_tt_t_new,x_at_t_new,x_et_t_new)
        
        # actions of the priority calculators
        cur_priorities_actions=(None,None)
        win_priorities_actions=(None,None)
        
        # set states in each service
        lightService.setStates(x_us_t_new, x_lr_t_new, x_le_t_new, x_ls_t_new, x_cur_t_new, MAX_LE)
        tempService.setStates(x_us_t_new, x_tr_t_new, x_te_t_new, x_cur_t_new, x_ac_t_new, x_win_t_new, MAX_TEMP,x_tt_t_new,x_et_t_new)
        airService.setStates(x_us_t_new, x_ar_t_new, x_ae_t_new, x_ap_t_new, x_win_t_new, x_cur_t_new, x_tr_t_new, x_te_t_new, x_at_t_new, x_et_t_new, MAX_CO2)
        
        # get reward values
        light_reward=lightService.getRewards(x_lr_new_t_2)
        light_reward_t=torch.tensor([[light_reward]],dtype=torch.float32,device=device)
        
        temp_reward=tempService.getReward(x_tr_new_t_2,x_tr_t)
        temp_reward_t=torch.tensor([[temp_reward]],dtype=torch.float32,device=device)
        
        air_reward=airService.getReward(x_ar_new_t_2,x_ar_t)
        air_reward_t=torch.tensor([[air_reward]],dtype=torch.float32,device=device)

        if light_reward>0:
            num_corr_light+=1

        print(f'num_corr_light: {num_corr_light}')

        if temp_reward>0:
            num_corr_temp+=1

        print(f'num_corr_temp: {num_corr_temp}')

        if air_reward>0:
            num_corr_air+=1

        print(f'num_corr_air: {num_corr_air}')

        if light_reward>0 and temp_reward>0 and air_reward>0:
            totalAcc[epoch,3]+=1

        print(f'three corr: {totalAcc[epoch,3]}')

        if step==steps-1:
            totalAcc[epoch,0]=num_corr_light
            totalAcc[epoch,1]=num_corr_temp
            totalAcc[epoch,2]=num_corr_air
        
        # cur_priorities_reward=(temp_reward,air_reward)
        # win_priorities_reward=(temp_reward,air_reward)
        
        # calculate the epoch total rewards
        totalEpochRewards[epoch,0]+=light_reward
        totalEpochRewards[epoch,1]+=temp_reward
        totalEpochRewards[epoch,2]+=air_reward

        # print(f'totalEpochRewards: {totalEpochRewards[epoch,:]}')
        
        # calculate the transitions
        light_transition=(x_light_t_norm,light_actions,light_reward_t,x_light_t_new_norm)
        temp_transition=(x_temp_t_norm,temp_actions,temp_reward_t,x_temp_t_new_norm)
        air_transition=(x_air_t_norm, air_actions, air_reward_t,x_air_t_new_norm)
        
        # cur_transition=(x_cur_p_t_norm,cur_priorities_actions,cur_priorities_reward,x_cur_p_t_new_norm)
        # win_transition=(x_win_p_t_norm,win_priorities_actions,win_priorities_reward,x_win_p_t_new_norm)
        
        lightService.updateReplayMemory(light_transition)
        tempService.updateReplayMemory(temp_transition)
        airService.updateReplayMemory(air_transition)
        
        # cur_calculator.updateReplayMemory(cur_transition)
        # win_calculator.updateReplayMemory(win_transition)
        
        # # services training process
        # _=lightService.train(epoch)
        # _=tempService.train(epoch)
        # _=airService.train(epoch)
        # # _=cur_calculator.train(epoch)
        # # _=win_calculator.train(epoch)

        rangeC=(x_t_pos_norm[0,0],x_t_pos_norm[0,0],
            x_t_pos_norm[0,1],x_t_pos_norm[0,1],
            x_t_pos_norm[0,2],x_t_pos_norm[0,2],
            x_t_pos_norm[0,3],x_t_pos_norm[0,3],
            x_t_pos_norm[0,4],x_t_pos_norm[0,4],
            x_t_pos_norm[0,5],x_t_pos_norm[0,5])

        transition_pos=np.zeros((1,30))

        transition_pos[0,0]=x_t_pos_norm[0,0].item()
        transition_pos[0,1]=x_t_pos_norm[0,1].item()
        transition_pos[0,2]=x_t_pos_norm[0,2].item()
        transition_pos[0,3]=x_t_pos_norm[0,3].item()
        transition_pos[0,4]=x_t_pos_norm[0,4].item()
        transition_pos[0,5]=x_t_pos_norm[0,5].item()
        transition_pos[0,6]=allActions[0].item()
        transition_pos[0,7]=allActions[1].item()
        transition_pos[0,8]=allActions[2].item()
        transition_pos[0,9]=allActions[3].item()
        transition_pos[0,10]=allActions[4].item()
        transition_pos[0,11]=allActions[5].item()
        transition_pos[0,12]=allActions[6].item()
        transition_pos[0,13]=allActions[7].item()
        transition_pos[0,14]=light_reward_t.item()
        transition_pos[0,15]=temp_reward_t.item()
        transition_pos[0,16]=air_reward_t.item()
        transition_pos[0,17]=rangeC[2].item()
        transition_pos[0,18]=rangeC[3].item()
        transition_pos[0,19]=rangeC[4].item()
        transition_pos[0,20]=rangeC[5].item()
        transition_pos[0,21]=rangeC[6].item()
        transition_pos[0,22]=rangeC[7].item()
        transition_pos[0,23]=rangeC[8].item()
        transition_pos[0,24]=rangeC[9].item()
        transition_pos[0,25]=rangeC[10].item()
        transition_pos[0,26]=rangeC[11].item()
        transition_pos[0,27]=MAX_LE
        transition_pos[0,28]=MAX_TEMP
        transition_pos[0,29]=MAX_CO2

        if ruldid==True:
            rules[0,epoch]+=1
        else:
            rules[2,epoch]+=1

        if existingRuldid==True:

            rules[4,epoch]+=1

        if not(light_reward>0 and temp_reward>0 and air_reward>0):

            if ruldid==True:

                # print(f'decrease distance and refine posRelayMemory!')

                MIN_LIGHT-=0.1
                MIN_TEMP-=0.01
                MIN_AIR-=0.1

                if MIN_LIGHT<=15:
                    MIN_LIGHT=15

                if MIN_TEMP<=1.5:
                    MIN_TEMP=1.5

                if MIN_AIR<=10:
                    MIN_AIR=10

                active_idx=np.where(posRelayMemory==active_rule)[0]
                posRelayMemory=np.delete(posRelayMemory,active_idx,axis=0).reshape(-1,30)






        if light_reward>0 and temp_reward>0 and air_reward>0:

            if ruldid==True:
                rules[1,epoch]+=1

            else:
                rules[3,epoch]+=1

            if existingRuldid==True:

                rules[5,epoch]+=1

            # print('reward>0')
            # print(f'len(posRelayMemory): {np.where(posRelayMemory[:,14]!=999)[0].shape[0]}')
            # print(f'=======')

            if posRelayMemory[0,14]==999:
                posRelayMemory[0,:]=transition_pos

            else:

                posRelayMemory=np.array(sorted(posRelayMemory, key=lambda x: x[0]),dtype=object)

                idx_notNone=np.where(posRelayMemory[:,14]!=999)[0]

                stateCandidiate=copy.deepcopy(posRelayMemory[idx_notNone,:6].reshape(-1,x_t_pos_norm.shape[1]))

                idx_keep=np.where(stateCandidiate[:,:1]==x_t_pos_norm.numpy()[:,:1])[0]

                if idx_keep.shape[0]!=0:

                    stateCandidiate=posRelayMemory[idx_keep,:6]
                    maxCandidate=posRelayMemory[idx_keep,-3:]
                    actionCandidiate=posRelayMemory[idx_keep,6:14]
                    transition_stateCandidate=copy.deepcopy(posRelayMemory[idx_keep,:])
                    posRelayMemory[idx_keep,:]=np.full((idx_keep.shape[0],30),999,dtype='float32')
                    posRelayMemory=np.array(sorted(posRelayMemory, key=lambda x: x[0]),dtype=object)
                    idx_notNone=np.where(posRelayMemory[:,14]!=999)[0]

                    stateCandidiate_air=stateCandidiate[:,[0,1]]
                    stateCandidiate_temp=stateCandidiate[:,[0,2,4]]
                    stateCandidiate_air=stateCandidiate[:,[0,3,5]]

                    x_t_pos_norm_light=x_t_pos_norm[:,[0,1]]
                    x_t_pos_norm_temp=x_t_pos_norm[:,[0,2,4]]
                    x_t_pos_norm_air=x_t_pos_norm[:,[0,3,5]]

                    stateCandidiate2_light=copy.deepcopy(stateCandidiate_light)
                    stateCandidiate2_light[:,0]/=4
                    stateCandidiate2_light[:,1]=np.divide(stateCandidiate2_light[:,1],maxCandidate[:,0])
                    # stateCandidiate2[:,2]/=maxCandidate[:,0]

                    x_t_pos_norm2_light=copy.deepcopy(x_t_pos_norm_light)
                    x_t_pos_norm2_light[0,0]/=4
                    x_t_pos_norm2_light[0,1]/=MAX_LE
                    # x_t_pos_norm2[0,2]/=MAX_LE


                    stateCandidiate2_temp=copy.deepcopy(stateCandidiate_temp)

                    stateCandidiate2_temp[:,0]/=4
                    stateCandidiate2_temp[:,1]/=maxCandidate[:,1]
                    stateCandidiate2_temp[:,2]=np.divide(stateCandidiate2_temp[:,2],maxCandidate[:,1])

                    x_t_pos_norm2_temp=copy.deepcopy(x_t_pos_norm_temp)

                    x_t_pos_norm2_temp[0,0]/=4
                    x_t_pos_norm2_temp[0,1]/=MAX_TEMP
                    x_t_pos_norm2_temp[0,2]/=MAX_TEMP

                    stateCandidiate2_air=copy.deepcopy(stateCandidiate_air)
                    stateCandidiate2_air[:,0]/=4
                    stateCandidiate2_air[:,1]/=maxCandidate[:,2]
                    stateCandidiate2_air[:,2]/=maxCandidate[:,2]

                    x_t_pos_norm2_air=copy.deepcopy(x_t_pos_norm_air)
                    x_t_pos_norm2_air[0,0]/=4
                    x_t_pos_norm2_air[0,1]/=MAX_CO2
                    x_t_pos_norm2_air[0,2]/=MAX_CO2



                    corr2_light=np.array([distance.cosine(np.array([i],dtype='float32'),x_t_pos_norm2_light) for i in stateCandidiate2_light]).reshape(-1)
                    corr2_euclidean_light=np.array([distance.euclidean(i,x_t_pos_norm_light) for i in stateCandidiate_light]).reshape(-1)

                    # print(f'corr2_light: {corr2_light}')
                    # print(f'corr2_euclidean_light: {corr2_euclidean_light}')
                    # print(f'=======')

                    corr_argsort_light=np.argsort(corr2_light).reshape(-1)
                    idx_corr_light=np.array([])

                    for i in corr_argsort_light:
                        if corr2_light[i]<0.0002 and corr2_euclidean_light[i]<MIN_LIGHT:
                            idx_corr_light=np.concatenate((idx_corr_light,np.array([i])))



                    corr2_temp=np.array([distance.euclidean(np.array([i],dtype='float32'),x_t_pos_norm2_temp) for i in stateCandidiate2_temp]).reshape(-1)
                    corr2_euclidean_temp=np.array([distance.euclidean(i,x_t_pos_norm_temp) for i in stateCandidiate_temp]).reshape(-1)

                    # print(f'corr2_temp: {corr2_temp}')
                    # print(f'corr2_euclidean_temp: {corr2_euclidean_temp}')
                    # print(f'=======')

                    corr_argsort_temp=np.argsort(corr2_temp).reshape(-1)
                    idx_corr_temp=np.array([])

                    for i in corr_argsort_temp:
                        if corr2_temp[i]<0.1 and corr2_euclidean_temp[i]<MIN_TEMP:
                            idx_corr_temp=np.concatenate((idx_corr_temp,np.array([i])))


                    corr2_air=np.array([distance.cosine(np.array([i],dtype='float32'),x_t_pos_norm2_air) for i in stateCandidiate2_air]).reshape(-1)
                    corr2_euclidean_air=np.array([distance.euclidean(i,x_t_pos_norm_air) for i in stateCandidiate_air]).reshape(-1)

                    # print(f'corr2_air: {corr2_air}')
                    # print(f'corr2_euclidean_air: {corr2_euclidean_air}')
                    # print(f'=======')

                    corr_argsort_air=np.argsort(corr2_air).reshape(-1)
                    idx_corr_air=np.array([])

                    for i in corr_argsort_air:
                        if corr2_air[i]<0.0002 and corr2_euclidean_air[i]<MIN_AIR:
                            idx_corr_air=np.concatenate((idx_corr_air,np.array([i])))


                    idx_corr_light=idx_corr_light.astype('int')
                    # print(f'idx_corr_light: {idx_corr_light}')
                    # print(f'=======')


                    idx_corr_temp=idx_corr_temp.astype('int')
                    # print(f'idx_corr_temp: {idx_corr_temp}')
                    # print(f'=======')

                    idx_corr_air=idx_corr_air.astype('int')
                    # print(f'idx_corr_air: {idx_corr_air}')
                    # print(f'=======')


                    idx_light_temp=idx_corr_light[np.in1d(idx_corr_light, idx_corr_temp)]
                    idx_light_temp_air=idx_light_temp[np.in1d(idx_light_temp,idx_corr_air)]

                    if idx_light_temp_air.shape[0]==0:

                        posRelayMemory[idx_notNone.shape[0]:(idx_notNone.shape[0]+idx_keep.shape[0]),:]=transition_stateCandidate
                        posRelayMemory[(idx_notNone.shape[0]+idx_keep.shape[0]),:]=transition_pos

                        del transition_stateCandidate

                    else:

                        act_idx=[]

                        diff_act_idx=[]

                        idx_corr=idx_light_temp_air

                        for i in idx_corr:

                            if transition_stateCandidate[i,6]==transition_pos[0,6] and \
                            transition_stateCandidate[i,7]==transition_pos[0,7] and \
                            transition_stateCandidate[i,8]==transition_pos[0,8] and \
                            transition_stateCandidate[i,9]==transition_pos[0,9] and \
                            transition_stateCandidate[i,10]==transition_pos[0,10] and \
                            transition_stateCandidate[i,11]==transition_pos[0,11] and \
                            transition_stateCandidate[i,12]==transition_pos[0,12] and \
                            transition_stateCandidate[i,13]==transition_pos[0,13]:
                                act_idx.append(i)

                            else:
                                diff_act_idx.append(i)

                        act_idx=np.array(act_idx).astype('int')
                        diff_act_idx=np.array(diff_act_idx).astype('int')

                        if np.array([sum(i[14:17])<sum(transition_pos[0,14:17]) for i in transition_stateCandidate[idx_corr]]).all():

                            transition_pos[0,1]=np.mean(np.concatenate((transition_stateCandidate[idx_corr,1].reshape(-1),transition_pos[0,1].reshape(-1))))
                            transition_pos[0,2]=np.mean(np.concatenate((transition_stateCandidate[idx_corr,2].reshape(-1),transition_pos[0,2].reshape(-1))))
                            transition_pos[0,3]=np.mean(np.concatenate((transition_stateCandidate[idx_corr,3].reshape(-1),transition_pos[0,3].reshape(-1))))
                            transition_pos[0,4]=np.mean(np.concatenate((transition_stateCandidate[idx_corr,4].reshape(-1),transition_pos[0,4].reshape(-1))))
                            transition_pos[0,5]=np.mean(np.concatenate((transition_stateCandidate[idx_corr,5].reshape(-1),transition_pos[0,5].reshape(-1))))

                            transition_pos[0,17]=min(np.concatenate((transition_stateCandidate[idx_corr,17].reshape(-1),transition_pos[0,17].reshape(-1))))
                            transition_pos[0,18]=max(np.concatenate((transition_stateCandidate[idx_corr,18].reshape(-1),transition_pos[0,18].reshape(-1))))
                            transition_pos[0,19]=min(np.concatenate((transition_stateCandidate[idx_corr,19].reshape(-1),transition_pos[0,19].reshape(-1))))
                            transition_pos[0,20]=max(np.concatenate((transition_stateCandidate[idx_corr,20].reshape(-1),transition_pos[0,20].reshape(-1))))
                            transition_pos[0,21]=min(np.concatenate((transition_stateCandidate[idx_corr,21].reshape(-1),transition_pos[0,21].reshape(-1))))
                            transition_pos[0,22]=max(np.concatenate((transition_stateCandidate[idx_corr,22].reshape(-1),transition_pos[0,22].reshape(-1))))
                            transition_pos[0,23]=min(np.concatenate((transition_stateCandidate[idx_corr,23].reshape(-1),transition_pos[0,23].reshape(-1))))
                            transition_pos[0,24]=max(np.concatenate((transition_stateCandidate[idx_corr,24].reshape(-1),transition_pos[0,24].reshape(-1))))
                            transition_pos[0,25]=min(np.concatenate((transition_stateCandidate[idx_corr,25].reshape(-1),transition_pos[0,25].reshape(-1))))
                            transition_pos[0,26]=max(np.concatenate((transition_stateCandidate[idx_corr,26].reshape(-1),transition_pos[0,26].reshape(-1))))

                            posRelayMemory[idx_notNone.shape[0],:]=transition_pos

                            del transition_stateCandidate

                        else:

                            if diff_act_idx.shape[0]!=0:

                                posRelayMemory[idx_notNone.shape[0]:(idx_notNone.shape[0]+diff_act_idx.shape[0]),:]=transition_stateCandidate[diff_act_idx,:]

                            if act_idx.shape[0]!=0:

                                transition_pos[0,1]=np.mean(np.concatenate((transition_stateCandidate[act_idx,1].reshape(-1),transition_pos[0,1].reshape(-1))))
                                transition_pos[0,2]=np.mean(np.concatenate((transition_stateCandidate[act_idx,2].reshape(-1),transition_pos[0,2].reshape(-1))))
                                transition_pos[0,3]=np.mean(np.concatenate((transition_stateCandidate[act_idx,3].reshape(-1),transition_pos[0,3].reshape(-1))))
                                transition_pos[0,4]=np.mean(np.concatenate((transition_stateCandidate[act_idx,4].reshape(-1),transition_pos[0,4].reshape(-1))))
                                transition_pos[0,5]=np.mean(np.concatenate((transition_stateCandidate[act_idx,5].reshape(-1),transition_pos[0,5].reshape(-1))))

                                transition_pos[0,17]=min(np.concatenate((transition_stateCandidate[act_idx,17].reshape(-1),transition_pos[0,17].reshape(-1))))
                                transition_pos[0,18]=max(np.concatenate((transition_stateCandidate[act_idx,18].reshape(-1),transition_pos[0,18].reshape(-1))))
                                transition_pos[0,19]=min(np.concatenate((transition_stateCandidate[act_idx,19].reshape(-1),transition_pos[0,19].reshape(-1))))
                                transition_pos[0,20]=max(np.concatenate((transition_stateCandidate[act_idx,20].reshape(-1),transition_pos[0,20].reshape(-1))))
                                transition_pos[0,21]=min(np.concatenate((transition_stateCandidate[act_idx,21].reshape(-1),transition_pos[0,21].reshape(-1))))
                                transition_pos[0,22]=max(np.concatenate((transition_stateCandidate[act_idx,22].reshape(-1),transition_pos[0,22].reshape(-1))))
                                transition_pos[0,23]=min(np.concatenate((transition_stateCandidate[act_idx,23].reshape(-1),transition_pos[0,23].reshape(-1))))
                                transition_pos[0,24]=max(np.concatenate((transition_stateCandidate[act_idx,24].reshape(-1),transition_pos[0,24].reshape(-1))))
                                transition_pos[0,25]=min(np.concatenate((transition_stateCandidate[act_idx,25].reshape(-1),transition_pos[0,25].reshape(-1))))
                                transition_pos[0,26]=max(np.concatenate((transition_stateCandidate[act_idx,26].reshape(-1),transition_pos[0,26].reshape(-1))))


                            posRelayMemory[(idx_notNone.shape[0]+diff_act_idx.shape[0]),:]=transition_pos

                            del transition_stateCandidate

                    # posRelayMemory=np.concatenate((posRelayMemory,np.full((idx_keep.shape[0],30),999,dtype='float32')),axis=0)


                else:

                    posRelayMemory[idx_notNone.shape[0],:]=transition_pos


        
        
        x_lr_t=copy.deepcopy(x_lr_t_new)
        x_tr_t=copy.deepcopy(x_tr_t_new)
        x_ar_t=copy.deepcopy(x_ar_t_new)
        
        x_cur_t=copy.deepcopy(x_cur_t_new)
        x_ls_t=copy.deepcopy(x_ls_t_new)
        x_ac_t=copy.deepcopy(x_ac_t_new)
        x_tt_t=copy.deepcopy(x_tt_t_new)
        x_et_t=copy.deepcopy(x_et_t_new)
        x_at_t=copy.deepcopy(x_at_t_new)
        x_win_t=copy.deepcopy(x_win_t_new)
        x_ap_t=copy.deepcopy(x_ap_t_new)

        print(f'rules: {rules[:,epoch]}')
        
        print(f'epoch:{epoch}, step:{step}, temp:{temp_reward_t}, air:{air_reward_t}, totalEpochRewards:{totalEpochRewards[epoch,:]}')

        print(f'=================================================================================')
        
        # %%
        
print(f'100 acc: {np.average(totalAcc,axis=0)}') 

print(f'100 lenRules: {np.average(np.average(lenRules,axis=0))}')       

#     if epoch%5==0:
#         lightService.targetLightServiceModel.load_state_dict(lightService.lightServiceModel.state_dict())
#         tempService.targetTempServiceModel.load_state_dict(tempService.tempServiceModel.state_dict())
#         airService.targetAirServiceModel.load_state_dict(airService.airServiceModel.state_dict())
             
#     if epoch%50==0 and epoch!=0:      
#         torch.save(lightService.lightServiceModel.state_dict(),f'data/lstm/lightService_lstm_v1_{1+int(epoch/50)}_eishomma_1.0_even_rep_existing_rules_2_c.pth')
#         torch.save(tempService.tempServiceModel.state_dict(),f'data/lstm/tempService_lstm_v1_{1+int(epoch/50)}_eishomma_1.0_even_rep_existing_rules_2_c.pth')
#         torch.save(airService.airServiceModel.state_dict(),f'data/lstm/airService_lstm_v1_{1+int(epoch/50)}_eishomma_1.0_even_rep_existing_rules_2_c.pth')
#         np.save(f'data/lstm/totalEpochRewards_lstm_v1_{1+int(epoch/50)}_eishomma_1.0_even_rep_existing_rules_2_c.npy',totalEpochRewards)
#         np.save(f'data/lstm/posReplayMemory_lstm_v1_{1+int(epoch/50)}_eishomma_1.0_even_rep_existing_rules_2_c.npy',posRelayMemory)
#         np.save(f'data/lstm/rules_lstm_v1_{1+int(epoch/50)}_eishomma_1.0_even_rep_existing_rules_2_c.npy',rules)
#         np.save(f'data/lstm/totalAcc_lstm_v1_{1+int(epoch/50)}_eishomma_1.0_even_rep_existing_rules_2_c.npy',totalAcc)
        

# torch.save(lightService.lightServiceModel.state_dict(),'data/lstm/lightService_lstm_v1_eishomma_1.0_even_rep_existing_rules_2_c.pth')
# torch.save(tempService.tempServiceModel.state_dict(),'data/lstm/tempService_lstm_v1_eishomma_1.0_even_rep_existing_rules_2_c.pth')
# torch.save(airService.airServiceModel.state_dict(),'data/lstm/airService_lstm_v1_eishomma_1.0_even_rep_existing_rules_2_c.pth')
np.save(f'data/lstm/totalEpochRewards_lstm_v1_eishomma_1.0_even_rep_existing_rules_2_c_100_4.npy',totalEpochRewards)


np.save(f'data/lstm/replayMemory_light_lstm_v1_eishomma_1.0_even_rep_existing_rules_2_c_100_4.npy',lightService.replayMemory)
np.save(f'data/lstm/replayMemory_temp_lstm_v1_eishomma_1.0_even_rep_existing_rules_2_c_100_4.npy',tempService.replayMemory)
np.save(f'data/lstm/replayMemory_air_lstm_v1_eishomma_1.0_even_rep_existing_rules_2_c_100_4.npy',airService.replayMemory)
np.save(f'data/lstm/posReplayMemory_lstm_v1_eishomma_1.0_even_rep_existing_rules_2_c_100_4.npy',posRelayMemory)
np.save(f'data/lstm/rules_lstm_v1_eishomma_1.0_even_rep_existing_rules_2_c_100_4.npy',rules)
np.save(f'data/lstm/lenRules_lstm_v1_eishomma_1.0_even_rep_existing_rules_2_c_100_4.npy',lenRules)
np.save(f'data/lstm/totalAcc_lstm_v1_eishomma_1.0_even_rep_existing_rules_2_c_100_4.npy',totalAcc)


# totalAcc=np.load(f'data/lstm/totalAcc_lstm_v1_eishomma_1.0_even_rep_existing_rules_2_c_100_4.npy',allow_pickle=True)

# print(f'100 acc: {np.average(totalAcc,axis=0)}') 
                
        
# lenRules=np.load(f'data/lstm/lenRules_lstm_v1_eishomma_1.0_even_rep_existing_rules_2_c_100_4.npy',allow_pickle=True)

# print(f'100 lenRules: {np.average(np.average(lenRules,axis=0))}')    


# posRelayMemory=np.load(f'data/lstm/posReplayMemory_lstm_v1_eishomma_1.0_even_rep_existing_rules_2_c_100_4.npy',allow_pickle=True)

# k_=0

# if posRelayMemory[0,14]!=999:
            
#     k_=(np.where(posRelayMemory[:,14]!=999)[0]).shape[0]
#     print(f'posReplayMemory size: {k_}')

#     print(lenRules[-1,-1])


# print('==========================================================================')

# print(posRelayMemory[:k_,:])

# print('==========================================================================') 

        
        
        
        
    
    
 
        

