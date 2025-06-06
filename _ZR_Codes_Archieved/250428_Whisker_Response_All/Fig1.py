'''
This script will show example location's all brain area response and whisker motion.

Fig 1B will show example case's response with motion

Fig 1C will show BV lag of all location 

Fig 1D will show max corr with motion,indicating coding to motion

Fig 1E will stat correlation between func area and motion.

'''

#%% import part here.
import Common_Functions as cf
import numpy as np
import matplotlib.pyplot as plt
from Brain_Atlas.Atlas_Mask import Mask_Generator
import seaborn as sns
import cv2
from tqdm import tqdm
import pandas as pd
from Atlas_Corr_Tools import *
import copy
from scipy.stats import pearsonr,ttest_ind
from Signal_Functions.Filters import *

all_path = cf.Get_Subfolders(r'D:\_DataTemp\OIS\All_WT_Datasets')
save_path = r'D:\_DataTemp\OIS\Motion_Midvars'
all_masks = cf.Load_Variable(save_path,'All_Masks.pkl')
all_corrs = cf.Load_Variable(save_path,'All_Corrs.pkl')
all_lags = cf.Load_Variable(save_path,'All_lags.pkl')

example_loc = all_path[0]
mask = np.load(cf.join(example_loc,'Mask.npy'))
height,width = mask.shape
z_series = np.load(cf.join(example_loc,'Z_Series_Full.npy'))
whiskers = np.load(cf.join(example_loc,'whisker_speed_5Hz.npy'))

n_frame,n_pix_postive = z_series.shape

#%% recover full-scale response and use ADT.
from Atlas_Corr_Tools import Atlas_Data_Tools
recovered_response = np.zeros(shape = (n_frame,height,width),dtype='f8')
recovered_response[:,mask==True] = z_series
ADT = Atlas_Data_Tools(recovered_response)
ADT.Get_All_Area_Response()
response_heats = ADT.Combine_Response_Heatmap()

#%%
# Fig 1B, 2 columns indication heatmap and 
start = 4000
end = 5000

fig,axes = plt.subplots(ncols=1,nrows=2,dpi=300,figsize = (8,6),gridspec_kw={'height_ratios': [1, 1]},sharex=True)
fontsize = 14
# plot area response
sns.heatmap(response_heats.loc[:,start:end],center = 0,ax = axes[0],xticklabels=False,cbar=False)
# plot averaged response
axes[1].plot(whiskers[start:end],label = 'Whisker Motion',alpha = 0.6)
axes[1].plot(z_series.mean(-1)[start:end]*2+1,label = 'Avr Response')
axes[1].legend(loc="upper left")
axes[1].set_yticks([])
fig.tight_layout()

#%% 
'''
Fig 1C, correlate bv response with whisker motion, getting the max lag and decay plot
'''
# define corr function.
def calculate_lagged_correlations(series_a, series_b, min_lag=-20, max_lag=20):

    series_a = pd.Series(series_a)
    series_b = pd.Series(series_b)
    correlations = []
    for lag in range(min_lag, max_lag + 1):
        # Shift series_b by the current lag (negative sign to align with standard definition)
        shifted_b = series_b.shift(-lag)
        # Compute Pearson correlation, ignoring NaN values
        corr = series_a.corr(shifted_b)
        correlations.append(corr)
    correlations = np.array(correlations)
    return correlations

lag_corrs = pd.DataFrame(columns = ['Mice','Lag','Corr'])
max_lag = np.zeros(len(all_path))
min_lag = -20
max_lag = 20
lag_range = np.arange(min_lag,max_lag+1)
for i,cloc in tqdm(enumerate(all_path)):
    mice = cloc.split('\\')[-1]
    c_motion = np.load(cf.join(cloc,'whisker_speed_5Hz.npy'))
    c_response = np.load(cf.join(cloc,'Z_Series_Full.npy')).mean(-1)
    corrs = calculate_lagged_correlations(c_motion,c_response,min_lag,max_lag)
    for j,clag in enumerate(lag_range):
        lag_corrs.loc[len(lag_corrs),:] = [mice,clag,corrs[j]]

#%% Plot part
fig,ax = plt.subplots(ncols=1,nrows=1,dpi=300,figsize = (4,4))
fontsize = 14
sns.lineplot(data = lag_corrs,x = 'Lag',y='Corr',ax =ax,c='black')
sns.lineplot(data = lag_corrs,x = 'Lag',y='Corr',ax =ax,hue='Mice',alpha = 0.15,legend=False)
ax.set_ylabel('Pearson R',fontsize=fontsize)
ax.set_xlabel('Lag',fontsize=fontsize)
ax.set_ylim(0,0.5)

# add max corr's lag here.
avr_corr = lag_corrs[['Lag','Corr']].groupby('Lag').mean()
max_lag = avr_corr.idxmax()['Corr']
ax.plot([max_lag,max_lag],[0,avr_corr.loc[max_lag]['Corr']],c='gray',linestyle='--')
#%%
'''
Fig 1D, getting each pix's correlation.
'''

all_masks = {}
all_corrs = {}
all_lags = {}
for i,cloc in enumerate(all_path):
    mice = cloc.split('\\')[-1]
    c_response = np.load(cf.join(cloc,'Z_Series_Full.npy'))
    c_motion = np.load(cf.join(cloc,'whisker_speed_5Hz.npy'))
    c_mask = np.load(cf.join(cloc,'Mask.npy'))
    _,pix_num = c_response.shape
    max_corrs = np.zeros(pix_num)
    max_lags = np.zeros(pix_num)
    for j in tqdm(range(pix_num)):
        c_pix = c_response[:,j]
        corr_plots = calculate_lagged_correlations(c_motion,c_pix,0,15)
        max_corrs[j]=corr_plots.max()
        max_lags[j]=corr_plots.argmax()
    all_masks[mice] = c_mask
    all_corrs[mice] = max_corrs
    all_lags[mice] = max_lags

cf.Save_Variable(save_path,'All_Masks',all_masks)
cf.Save_Variable(save_path,'All_Corrs',all_corrs)
cf.Save_Variable(save_path,'All_lags',all_lags)

#%% recover all location's map and getting combined mask.
def Recover_Map(vector,mask):
    graph = np.zeros(mask.shape)
    graph[mask==True] = vector
    return graph
all_mice = list(all_masks.keys())

# test example location for bug fix.
# test = Recover_Map(all_corrs[all_mice[id]],all_masks[all_mice[id]])
# sns.heatmap(test,square=True)
# print(test.max())

## cycle all location for average graph.
joint_mask = np.ones(shape = (330,285))
all_corr_graphs = np.zeros(shape = (len(all_mice),330,285))
all_lag_graphs = np.zeros(shape = (len(all_mice),330,285))
for i,c_mice in enumerate(all_mice):
    joint_mask *= all_masks[c_mice] # use only joint masks
    c_corr = Recover_Map(all_corrs[c_mice],all_masks[c_mice])
    c_lag = Recover_Map(all_lags[c_mice]-all_lags[c_mice].min(),all_masks[c_mice])
    all_corr_graphs[i,:,:] = c_corr
    all_lag_graphs[i,:,:] = c_lag

MG = Mask_Generator(bin=4)
area_boulders = MG.Area_Counters()
#%% Plot averaged corr graph.
# Fig 1D is generated here.
avr_max_corr = all_corr_graphs.mean(0)*joint_mask
avr_max_corr += area_boulders/255

fig,ax = plt.subplots(ncols=1,nrows=1,dpi=300,figsize = (4,4))
fontsize = 14
sns.heatmap(data = avr_max_corr,square=True,xticklabels=False,yticklabels=False,vmax =0.45,center=0.35,vmin = 0.25,cbar=False,ax = ax,cmap="rocket")


#%% stat of motion corr by loc
# binary high&low corr brain area.
all_corr_stat_frame = pd.DataFrame(index=range(1000000),columns=['Mice','Correlation'])
counter=0
for i,c_mice in enumerate(all_mice):
    c_mice_corrs = all_corrs[c_mice]
    for j,c_corr in tqdm(enumerate(c_mice_corrs)):
        all_corr_stat_frame.loc[counter]=[c_mice,c_corr]
        counter+=1
all_corr_stat_frame = all_corr_stat_frame.dropna(axis=0,how='any')
#%% Plot histogram of correlation value, and bifurcate data into high and low motion related.
fig,ax = plt.subplots(ncols=1,nrows=1,dpi=300,figsize = (4,4))
sns.histplot(data=all_corr_stat_frame,hue='Mice',x='Correlation',ax = ax,alpha=0.5)


