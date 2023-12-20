import torch 
import random
import numpy as np

from astropy import modeling

from configurations import *


# %%
random.seed(0)

torch.autograd.set_detect_anomaly(True)

device=torch.device('cpu')
# device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# %%
# # common states
# # user state
# X_US=torch.tensor([[0,1,2,3]],dtype=torch.float32,device=device)
# lenUS=X_US.shape[1]
# # %% light service
# # lamp state
# X_LS=torch.tensor([[0,1,2,3,4]],dtype=torch.float32,device=device)
# lenLS=X_LS.shape[1]
# # curtain state
# X_CUR=torch.tensor([[0,1/2,1]],dtype=torch.float32, device=device)
# lenCUR=X_CUR.shape[1]
# # %% temperature service
# X_AC=torch.tensor([[15,16,17,18,19,20,21,22,23,24,25,26]], dtype=torch.float32,device=device)
# # %% air quality service
# X_AP=torch.tensor([[0,1,2,3,4,5]],dtype=torch.float32,device=device)
# lenAP=X_AP.shape[1]

# L_CMH=torch.tensor([[0,60,170,280,390,500]],dtype=torch.float32,device=device)

# E_WH=torch.tensor([[0,10.1,13.5,19.2,35.8,56.8]])
# # %%
# # neural network settings
# batch_size=120
# num_epochs=10000
steps=288

# learning_rate=0.001
# epsilon=0.1

# max_x_light=1005

# # %%
# # user changes simulation
# def X_us_t_generation():
#     x_us_t=torch.randint(0,4,(1,1)).to(device)
#     return x_us_t

# %%
# the outdoor light intensity simulation
# the maximum outdoor light intensity is 605
def intensity_simulator(max_x_light2=1000):


    m = modeling.models.Gaussian1D(amplitude=max_x_light2, mean=12, stddev=3)
    x = np.linspace(0, 24, steps)
    data = m(x)
    data = data + 5*np.random.random(x.size)
    data=data.astype(np.float32)
    
    x_le_t=torch.from_numpy(data).reshape((1,-1))
    
    x_le_t=x_le_t.to(device)

    # print(f'x_le_t: {x_le_t.shape}')
    return x_le_t

# # %%
# # the outdoor temperature simulation
# def x_te_t_generation(max_x_temps=27):
#     m = modeling.models.Gaussian1D(amplitude=max_x_temps, mean=15, stddev=5)
#     x = np.linspace(0, 24, 288)
#     data = m(x)
#     data = data + 3*np.random.random(x.size)
#     data=data.astype(np.float32)
#     x_te_t=torch.from_numpy(data).reshape((1,-1))
    
#     return x_te_t
# %%


