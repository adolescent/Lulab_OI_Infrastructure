'''
This scrip will try to reduce neuro-response to a 3D neuro trace series.
I'll try PCA-UMAP dimension reduction.

'''


#%%

#%%
import Common_Functions as cf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm
import pandas as pd
from Brain_Atlas.Atlas_Mask import Mask_Generator
from Signal_Functions.Pattern_Tools import Do_PCA
import seaborn as sns
import umap
import copy
  
wp = r'D:\_DataTemp\OIS\Wild_Type\Preprocessed'
MG = Mask_Generator(bin=4)


on_response,on_mask = cf.Load_Variable(wp,'Response_1d.pkl')


#%% recover graph from pc.
# Do PCA
PC_Comp_raw,coords_raw,model_raw = Do_PCA(on_response,feature='Area',pcnum=50)
recovered_series = np.dot(coords_raw,PC_Comp_raw) # NOTE the shape is different!
#%% UMAP transfer 

reducer = umap.UMAP(n_components=3,n_neighbors=300,min_dist=0.02)
reducer.fit(recovered_series)# N_Sample*N_Feature
u = reducer.transform(recovered_series)

plt.scatter(u[:5000,0],u[:5000,1],c = range(len(u[:5000])),s=3,edgecolor='none',alpha = 0.7,cmap='jet')


#%% Plot umap result to 3d 
from matplotlib.animation import FuncAnimation
from PIL import Image  # For GIF processing

# u = coords_raw[:,[0,1,2]] # for spatial PCA
# u = PC_Comp_raw_t[[0,1,2],:]  # for temporal PCA
start = 10000
end = 13000
plotable = u[start:end,:]


fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': '3d'},figsize = (10,8),dpi = 180)
# ax.view_init(elev=45, azim=200)
ax.grid(False)
ax.view_init(elev=25, azim=70)
scatter = ax.scatter(plotable[:,0],plotable[:,1],plotable[:,2],c=range(len(plotable)), cmap='jet', alpha=0.5,edgecolors='none',s=5) # for spatial PCA
# scatter = ax.scatter(u[0,:],u[1,:],u[2,:],c=range(len(u[0])), cmap='rainbow', alpha=0.5,edgecolors='none',s=5) # for temporal PCA
# scatter = ax.plot(u[:,0],u[:,1],u[:,2])

# plt.colorbar(scatter, label='Color Scale')
# ax.legend(['Data Points'])
# ax.set_xlim(-100,100)
# ax.set_ylim(-40,40)
# ax.set_zlim()
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('UMAP 3')


'''
We can find significant local cluster of neuro-traces. So HMM here might be possible.
'''
# #%% recover part
# c_vec = reoverd_series[:,12324]

# recover = copy.deepcopy(on_mask).astype('f8')
# recover[recover==True] = c_vec
# sns.heatmap(recover,center = 0)


#%%###############################################
# HMM test

import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

# Load your data (shape: 1400x18000)
data = on_response  # Replace with your data
X = data.T  # Reshape to (18000, 1400)

# Standardize features (recommended)
scaler = StandardScaler() # z score
X_scaled = np.clip(scaler.fit_transform(X),-3,3)
# X_scaled = X_scaled/3
#
import os
os.environ["OMP_NUM_THREADS"] = '1.'
# Initialize the HMM
model = hmm.GaussianHMM(n_components=12, covariance_type="diag",n_iter = 1000,tol=0.001)
# model = hmm.GMMHMM(n_components=15, covariance_type="diag",n_iter = 10000,tol=0.0001)

# Fit the model
model.fit(X_scaled)
# Predict the hidden states for each time step
hidden_states = model.predict(X_scaled)
# a = model.predict_proba(X_scaled)
cf.Save_Variable(wp,'HMM_Model',model)
# hidden_states.shape = (18000,)
#%% show hidden state
# Mean vectors (shape: 6x1400)
model = cf.Load_Variable(wp,'HMM_Model.pkl')
all_mats = model.means_
# all_mats = model.means_[:,0,:] # for GMMHMM
# Covariance matrices (shape: 6x1400)
state_covariance_matrices = model.covars_

# Transition matrix (shape: 6x6)
transition_matrix = model.transmat_
# sns.heatmap(transition_matrix)
#%% test recover
# c_vec =on_response[:,hidden_states==3].mean(1)
c_vec = all_mats[4,:]
# c_vec = state_covariance_matrices[6,0,:]

recover = copy.deepcopy(on_mask).astype('f8')
recover[recover==True] = c_vec
sns.heatmap(recover,square=True)
# sns.heatmap(recover,square=True,center = -0.6,vmin=-0.9,vmax = -0.3)


#%%
# sample new data from this hmm.
X, Z = model.sample(50)
recover = copy.deepcopy(on_mask).astype('f8')
recover[recover==True] = X[0,:]
sns.heatmap(recover,square=True)
#%%
# well , it seems not good enough. Let's try something different.
# area_resp = np.zeros(shape = ())
series = np.load(cf.join(wp,'z_series.npy'))
mask = cv2.imread(cf.join(wp,'Chamber_mask.png'),0)>0
mask = mask*mask[:,::-1]
joint_mask = (series.std(0)>0)*mask
series = series*joint_mask

# a = MG.Avr_By_Area(series[0,:,:])
area_resp = np.zeros(shape = (1802,42),dtype = 'f8')
for i in tqdm(range(1802)):
    area_resp[i,:] = np.array(MG.Avr_By_Area(series[i,:,:])['Response'])

# del series
#%%
scaler = StandardScaler() # z score
X_scaled = np.clip(scaler.fit_transform(area_resp),-3,3)
# X_scaled = X_scaled/3
#
import os
os.environ["OMP_NUM_THREADS"] = '1.'
# Initialize the HMM
model = hmm.GaussianHMM(n_components=7, covariance_type="diag",n_iter = 10000,tol=0.001)
# model = hmm.GMMHMM(n_components=15, covariance_type="diag",n_iter = 10000,tol=0.0001)

# Fit the model
model.fit(X_scaled)
# Predict the hidden states for each time step
hidden_states = model.predict(X_scaled)
all_mats = model.means_

#%%
a = MG.Avr_By_Area(series[0,:,:])
X, Z = model.sample(150)
sns.heatmap(X)

#%% plot recovered graph
# c_map = MG.Get_Weight_Map(a['Name'],X[0,:])
c_map = MG.Get_Weight_Map(a['Name'],all_mats[4,:])
sns.heatmap(c_map,square=True,center=0)