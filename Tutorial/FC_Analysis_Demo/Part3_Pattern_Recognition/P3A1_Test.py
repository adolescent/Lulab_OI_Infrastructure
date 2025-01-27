'''
Developing codes for P3A1 PCA base analysis.

We will do :
1. Direct heamap visualization
2. Area-based PCA analysis (weight included)
3. Mosiac avraged PCA analysis


'''

#%% Load and import 

import Common_Functions as cf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm
import pandas as pd
from Brain_Atlas.Atlas_Mask import Mask_Generator
from Atlas_Corr_Tools import *
from scipy.stats import pearsonr
from Signal_Functions.Pattern_Tools import Do_PCA

wp = r'D:\_DataTemp\OIS\Wild_Type\Preprocessed'
series = np.load(cf.join(wp,'z_series.npy'))

# join chamber mask with brain area mask, getting only mask with values.
mask = cv2.imread(cf.join(wp,'Chamber_mask.png'),0)>0
joint_mask = (series.std(0)>0)*mask

# mask and clip input graph.
# NOTE this part is important for getting correct results.
series = np.clip(series,-3,3)*joint_mask
MG = Mask_Generator(bin=4)


#%%
'''
Work 0:
    Get area averaged response of all brain areas.
    You can see area response heatmap directly.

'''
from Atlas_Corr_Tools import Atlas_Data_Tools
ADT = Atlas_Data_Tools(series=series,bin=4,min_pix=100)
ADT.Get_All_Area_Response(keep_unilateral=False)
Area_Response = ADT.Area_Response
Area_Response_Heatmap = ADT.Combine_Response_Heatmap()

#%%
'''
Do Spacial PCA, use area as feature, plot most explained vars.
'''

# input must be in shape N_feature*N_Sample
area_namelist = list(Area_Response_Heatmap.index)
PC_Comp,coords,model = Do_PCA(Area_Response_Heatmap,feature='Area',pcnum=20)
plt.plot(model.explained_variance_ratio_[:10])
plt.ylim(0,0.2)

# we can see weight of component is varying and not correlated with each other.
# sns.heatmap(coords.T[:,5000:6000],center = 0,vmax=2,vmin = -2)
#%% Visualize PC comp weights
# c_comp = PC_Comp[4,:]
c_comp = PC_Comp[5,:]
c_comp_visual = MG.Get_Weight_Map(area_namelist,c_comp)
sns.heatmap(c_comp_visual,center=0,square=True)

#%%
'''
Do time PCA, use time as features, and show different networks
'''
PC_Comp_t,coords_t,model_t = Do_PCA(Area_Response_Heatmap,feature='Time',pcnum=20)
# plt.plot(model_t.explained_variance_ratio_[:10])
# plt.ylim(0,0.2)

# plot example 
sns.heatmap(PC_Comp_t[:,5000:6000],center = 0)

#%%
c_comp = coords_t[:,0]
c_comp_visual = MG.Get_Weight_Map(area_namelist,c_comp)
sns.heatmap(c_comp_visual,center=0,square=True)

#%%
'''
Try to do raw PCA for graph.
NOTE DO NOT Run this unless you have BIG memory, otherwise it will cost hours of times, even stuck.(Server recommended)
'''
# bin graph for faster calculation.
series_reshape = series.reshape(18025,66,5,57,5)
# to solve boulder, we cannot mean directly.
series_bin = np.zeros(shape = (18025,66,57),dtype='f8')
mask_reshape = (series.std(0)>0).reshape(66,5,57,5)
for i in tqdm(range(66)):
    for j in range(57):
        c_block = series_reshape[:,i,:,j,:].sum(-1).sum(-1)
        c_masknum = mask_reshape[i,:,j,:].sum()
        if c_masknum>0:
            series_bin[:,i,j] = c_block/c_masknum


plt.imshow(series_bin[5110,:,:])
#%% Run PCA
on_mask = series_bin.std(0)>0
on_mask *= on_mask[:,::-1]
on_response = series_bin[:,on_mask==True].T

PC_Comp_raw,coords_raw,model_raw = Do_PCA(on_response[:,:2000],feature='Area',pcnum=75)

sns.heatmap(coords_raw[:,:10].T,center=0,vmax = 10,vmin = -10)

#%% fill back the graph to mask.
recover = copy.deepcopy(on_mask).astype('f8')
recover[recover==True] = PC_Comp_raw[0,:]
sns.heatmap(recover,center=0,square=True)

#%% ################################
# If you want to test on time series.
PC_Comp_raw_t,coords_raw_t,model_raw_t = Do_PCA(on_response[:,:2000],feature='Time',pcnum=75)
sns.heatmap(PC_Comp_raw_t[:20,:],center=0,vmax = 0.15,vmin = -0.15)


#%% fill back the graph to mask.
recover = copy.deepcopy(on_mask).astype('f8')
recover[recover==True] = coords_raw_t[:,0]
sns.heatmap(recover,center=0,square=True)

#%% Fill back PC comps, and avr it on brain-area map.

exp = np.zeros(shape = (330,285),dtype='f8')

for i in range(66):
    for j in range(57):
        c_block = recover[i,j]
        exp[i*5:(i+1)*5,j*5:(j+1)*5]=c_block

a = MG.Avr_By_Area(graph=exp,min_pix=30)
recover_avr = MG.Get_Weight_Map(area_names=a['Name'],weight_frame=a['Response'])

fig,ax = plt.subplots(nrows=1,ncols=2,figsize = (12,7),dpi = 180)
sns.heatmap(recover,center=0,square=True,ax = ax[0],cbar = False)
sns.heatmap(recover_avr,center=0,square=True,ax = ax[1],cbar=False)

