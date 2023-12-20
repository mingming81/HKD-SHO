# extract_data, deal with outliers
# https://www.askpython.com/python/examples/detection-removal-outliers-in-python
# https://www.askpython.com/python/examples/impute-missing-data-values
# https://machinelearningmastery.com/knn-imputation-for-missing-values-in-machine-learning/
# interpolate to obtain data every 5 minutes
# https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html#id3
# %%
import torch 

import numpy as np

import ast

import pandas as pd

import sys

# sys.path.append(".")

from configurations import *
from sklearn.impute import SimpleImputer
# %%

device=torch.device('cpu')
# device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# %%

# filename='data/co2_smo_surface-insitu_1_ccgg_HourlyData.txt'

# data=np.loadtxt(filename,delimiter=' ',dtype=str)

# pddata=pd.read_csv(filename,delimiter=" ")

# %%
# keys=data[[0],:]

# hour=data[1:,[4]].astype(float)
# co2=data[1:,[8]].astype(float)

# npdata=np.zeros((394488,2))
# npdata[:,0]=hour[:,0]
# npdata[:,1]=co2[:,0]
# %%

# pdco2=pd.DataFrame(list(npdata),columns=['hour','co2'])


# %%
# # %%
import matplotlib.pyplot as plt
# plt.boxplot(dic["co2"])

# pdco2.boxplot(['co2'])

# %% remove the outliers
# for x in ["co2"]:
#     q75,q25=np.percentile(pdco2.loc[:,x],[75,25])
#     intr_qr=q75-q25
    
#     max=q75+(1.5*intr_qr)
#     min = q25-(1.5*intr_qr)
    
#     pdco2.loc[pdco2[x]<min,x]=np.nan
#     pdco2.loc[pdco2[x]>max,x]=np.nan

# %%
# print(pdco2.isnull().sum())


# %% interpolation with KNN

import os
from sklearn.impute import KNNImputer



# imputer=KNNImputer(n_neighbors=3, weights='distance', metric='nan_euclidean')
# pdco2=pd.DataFrame(imputer.fit(pdco2).transform(pdco2),columns=pdco2.columns)



# pdco2.to_pickle("./data/pdco2.pkl")

# %%
import pickle

# pdco22=pd.read_pickle("./data/pdco2.pkl")

with open("./../../data/pdco2.pkl","rb") as fh:
    pdco2=pickle.load(fh)

pdco2=pdco2.replace(-999.99, np.nan)
fill_NaN = SimpleImputer(missing_values=np.nan, strategy='mean')
pdco22 = pd.DataFrame(fill_NaN.fit_transform(pdco2))

pdco22.columns = pdco2.columns
pdco22.index = pdco2.index

# %%
# print(pdco2.isnull().sum())

# %% interpolate data

from scipy.interpolate import interp1d


def co2_simulator():
    
    np.random.seed(0)

    x=np.linspace(0,int(24*1)-1,num=int(24*1),endpoint=True)
    y=np.array(pdco22["co2"])[-int(24*1):]
    
    # f1= interp1d(x, y)
    
    
    f2=interp1d(x,y,kind='cubic')
    
    xnew=np.linspace(0,int(24*1)-1,num=int(60/5*24*1),endpoint=True)
    # y1=f1(xnew)
    y2=f2(xnew)+np.random.normal(size=xnew.size)
    
    x_air_t=torch.from_numpy(y2).reshape((1,-1))
    
    x_air_t=x_air_t.to(device)
    
    return x_air_t
 
# %%
# from scipy.interpolate import interp1d

# np.random.seed(0)

# fig,ax = plt.subplots(figsize = (7,4)) 

# x=np.linspace(0,int(24*1)-1,num=int(24*1),endpoint=True)
# y=np.array(pdco2["co2"])[-int(24*1):]

# f1= interp1d(x, y)


# f2=interp1d(x,y,kind='cubic')

# xnew=np.linspace(0,int(24*1)-1,num=int(60/5*24*1),endpoint=True)
# y1=f1(xnew)
# y2=f2(xnew)+np.random.normal(size=xnew.size)

# ax.plot(x,y,'o',xnew,y1,'*',xnew,y2,'--')


# plt.grid()
# plt.legend(['data','linear','cubic'])
# plt.show()

# # # conclusion: finally, f1 performs better than f2
# %%
# user sarima to adapt to the data














