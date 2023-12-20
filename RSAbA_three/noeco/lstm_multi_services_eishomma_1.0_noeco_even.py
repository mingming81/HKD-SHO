
import torch 
import numpy as np
import pandas as pd

from collections import deque

import random

from simulated_light import *
from simulated_temp_3_2_tanh import *
from simulated_co2 import *

from LSTM_LIGHT_SERVICE_8_NOECO_CELL import *
from LSTM_AIR_SERVICE_RANGE_8_v1_NOECO_CELL_EISHOMMA_1_0_c import *
from LSTM_TEMP_SERVICE_88_NOECO_CELL_EISHOMMA_1_0 import *

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

# X_AT=torch.tensor([[i/10 for i in range(0,50)]],dtype=torch.float32,device=device)
# X_AT=X_T[[0],1:]

X_TET=torch.tensor([[i/10 for i in range(0,50)]],dtype=torch.float32,device=device)

X_AET=torch.tensor([[i/10 for i in range(0,50)]],dtype=torch.float32,device=device)

X_ET=copy.deepcopy(X_AET)

X_AP=torch.tensor([[0,1,2,3,4,5]],dtype=torch.float32,device=device)

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
    
# %%

lightService=LightService_cell()
# lightService=LightService_cell()
lightService.lightServiceModel.load_state_dict(torch.load('data/lstm/lightService_lstm_v1_5_eishomma_1.0_noeco_even.pth'))
lightService.targetLightServiceModel.load_state_dict(torch.load('data/lstm/lightService_lstm_v1_5_eishomma_1.0_noeco_even.pth'))

tempService=TempService_cell()
tempService.tempServiceModel.load_state_dict(torch.load('data/lstm/tempService_lstm_v1_5_eishomma_1.0_noeco_even.pth'))
tempService.targetTempServiceModel.load_state_dict(torch.load('data/lstm/tempService_lstm_v1_5_eishomma_1.0_noeco_even.pth'))

airService=AirService_cell()
airService.airServiceModel.load_state_dict(torch.load('data/lstm/airService_lstm_v1_5_eishomma_1.0_noeco_even.pth'))
airService.targetAirServiceModel.load_state_dict(torch.load('data/lstm/airService_lstm_v1_5_eishomma_1.0_noeco_even.pth'))

# cur_calculator=CalculationPriorities_cell(num_inputs=X_US.shape[1]+1+1+1,num_outputs=3)
# cur_calculator.calPrioModel.load_state_dict(torch.load('data/structure1/cur_calculator_v4.pth'))
# cur_calculator.targetCalPrioModel.load_state_dict(torch.load('data/structure1/cur_calculator_v4.pth'))

# win_calculator=CalculationPriorities_cell(num_inputs=X_US.shape[1]+1+1,num_outputs=2)
# win_calculator.calPrioModel.load_state_dict(torch.load('data/structure1/win_calculator_v4.pth'))
# win_calculator.targetCalPrioModel.load_state_dict(torch.load('data/structure1/win_calculator_v4.pth'))

seenTrainingStates=deque(maxlen=100000)

totalEpochRewards=np.full([num_epochs,3],0)

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
    
    X_us_t=user_simulator()
    
    X_le_t=intensity_simulator()
    idx_x_le_t=(torch.max(X_le_t,dim=1,keepdim=True)[1]).to(device)
    MAX_LE=X_le_t[[[0]],idx_x_le_t.item()]
    MAX_LE=int(MAX_LE)+1
    
    X_te_t=temp_simulator()
    idx_x_te_t=(torch.max(X_te_t,dim=1,keepdim=True)[1]).to(device)
    MAX_TEMP=X_te_t[[[0]],idx_x_te_t.item()]
    MAX_TEMP=int(MAX_TEMP)+1
    
    
    X_ae_t=co2_simulator()
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
        
        
        
        if step==0:
            # user states
            x_us_t=X_us_t[0,step]
            x_us_t=x_us_t.reshape(1,-1)
   
            
        else:
            if step%12==0:
                x_us_t=user_simulator()
            x_us_t=x_us_t.reshape((1,-1))
            
            
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
        
        
        # x_cur_p_t_norm=torch.cat((x_us_t_norm,x_le_t_norm,x_te_t_norm,x_ae_t_norm),axis=1)
        
        
        
        # x_win_p_t_norm=torch.cat((x_us_t_norm,x_te_t_norm,x_ae_t_norm),axis=1)
        
        
        
        sigma=torch.rand(1).item()
      
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

        print(f'x_us_t: {x_us_t}')

        print(f'x_le_t: {x_le_t}')
        print(f'x_lr_t: {x_lr_t}')
        print(f'x_lr_t_new: {x_lr_t_new}')
        print(f'x_lr_new_t_2: {x_lr_new_t_2}')
        

        print(f'x_te_t: {x_te_t}')
        print(f'x_tr_t: {x_tr_t}')
        print(f'x_tr_t_new: {x_tr_t_new}')
        print(f'x_tr_new_t_2: {x_tr_new_t_2}')

        print(f'x_ae_t: {x_ae_t}')
        print(f'x_ar_t: {x_ar_t}')
        print(f'x_ar_t_new: {x_ar_t_new}')
        print(f'x_ar_new_t_2: {x_ar_new_t_2}')

        
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
        
        # services training process
        _=lightService.train(epoch)
        _=tempService.train(epoch)
        _=airService.train(epoch)
        # _=cur_calculator.train(epoch)
        # _=win_calculator.train(epoch)
        
        
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
        
        print(f'epoch:{epoch}, step:{step}, temp:{temp_reward_t}, air:{air_reward_t}, totalEpochRewards:{totalEpochRewards[epoch,:]}')

        print(f'=================================================================================')
        
        # %%
        
    
    if epoch%5==0:
        lightService.targetLightServiceModel.load_state_dict(lightService.lightServiceModel.state_dict())
        tempService.targetTempServiceModel.load_state_dict(tempService.tempServiceModel.state_dict())
        airService.targetAirServiceModel.load_state_dict(airService.airServiceModel.state_dict())
             
    if epoch%200==0 and epoch!=0:      
        torch.save(lightService.lightServiceModel.state_dict(),f'data/lstm/lightService_lstm_v1_{1+5+int(epoch/100)}_eishomma_1.0_noeco_even.pth')
        torch.save(tempService.tempServiceModel.state_dict(),f'data/lstm/tempService_lstm_v1_{1+5+int(epoch/200)}_eishomma_1.0_noeco_even.pth')
        torch.save(airService.airServiceModel.state_dict(),f'data/lstm/airService_lstm_v1_{1+5+int(epoch/200)}_eishomma_1.0_noeco_even.pth')
        np.save(f'data/lstm/totalEpochRewards_lstm_v1_{1+5+int(epoch/200)}_eishomma_1.0_noeco_even.npy',totalEpochRewards)
        

torch.save(lightService.lightServiceModel.state_dict(),'data/lstm/lightService_lstm_v1_eishomma_1.0_noeco_even.pth')
torch.save(tempService.tempServiceModel.state_dict(),'data/lstm/tempService_lstm_v1_eishomma_1.0_noeco_even.pth')
torch.save(airService.airServiceModel.state_dict(),'data/lstm/airService_lstm_v1_eishomma_1.0_noeco_even.pth')
np.save(f'data/lstm/totalEpochRewards_lstm_v1_eishomma_1.0_noeco_even.npy',totalEpochRewards)

        
        
        
        
        
    
    
 
        

