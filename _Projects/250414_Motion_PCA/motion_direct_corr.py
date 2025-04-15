'''
This script will just do lagged PCA between raw motion and each pix response.
'''


#%%
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

# load frame
wp = r'D:\ZR\_Data_Temp\Ois200_Data\activity_analysis\47#'
series = np.load(cf.join(wp,'z_series.npy'))
mask = cv2.imread(cf.join(wp,'chambermask.png'),0)
mask = mask*(series.std()>0)

# filt and calculate average speed.
avr_speed = np.load(cf.join(wp,'whisker_speed_5Hz.npy'))
avr_speed = Signal_Filter_1D(avr_speed,False,0.5,5)

#%%
'''
Step 1, define lagged correlation method.
'''

def calculate_lagged_correlations(series_a, series_b, min_lag=-20, max_lag=20):
    correlations = []
    for lag in range(min_lag, max_lag + 1):
        # Shift series_b by the current lag (negative sign to align with standard definition)
        shifted_b = series_b.shift(-lag)
        # Compute Pearson correlation, ignoring NaN values
        corr = series_a.corr(shifted_b)
        correlations.append(corr)
    correlations = np.array(correlations)
    return correlations

lagged_corrs = calculate_lagged_correlations(pd.Series(avr_speed), pd.Series(series.mean(-1).mean(-1)))
fig,ax = plt.subplots(ncols=1,nrows=1,figsize = (4,4),dpi=240)
ax.plot(lagged_corrs)
ax.axvline(x = lagged_corrs.argmax(),linestyle='--',color='gray')

ax.set_xticks(np.arange(0,40,4))
ax.set_xticklabels(np.arange(-20,20,4))
#%% 
'''
Step 2, for all pix inside mask, calculate it's lag and best corr
'''
data_1d = series[:,mask>0].T

best_corrs = np.zeros(len(data_1d))
corr_lag = np.zeros(len(data_1d))
lagged_corr_plots = np.zeros(shape = (len(data_1d),31))

for i in tqdm(range(len(data_1d))):
    c_resp = data_1d[i,:]
    lagged_corrs = calculate_lagged_correlations(pd.Series(avr_speed), pd.Series(c_resp),min_lag=-10, max_lag=20)
    lagged_corr_plots[i,:] = lagged_corrs
    best_corrs[i] = lagged_corrs.max()
    corr_lag[i] = np.where(lagged_corrs == lagged_corrs.max())[0][0]
#%% recover and plot 
    
rec_graph = np.zeros(mask.shape)
rec_graph[mask>0] = corr_lag
# rec_graph[mask>0] = point_coords[:,0]
sns.heatmap(rec_graph,center = 16,vmax = 20,vmin = 12,square = True,xticklabels = False,yticklabels = False,cmap='rainbow')

#%% 
'''
Step 3, corr by time lag, making it possible to see network ongoing.
'''
rec_graph = np.zeros(mask.shape)
rec_graph[mask>0] = lagged_corr_plots[:,7]
# rec_graph[mask>0] = best_corrs
sns.heatmap(rec_graph,center = 0,square = True,xticklabels = False,yticklabels = False)

#%%
fig,ax = plt.subplots(ncols = 5,nrows=6,dpi = 300,figsize = (8,12))

for i in range(31):
    rec_graph = np.zeros(mask.shape)
    rec_graph[mask>0] = lagged_corr_plots[:,i]
    sns.heatmap(rec_graph,center = 0.3,square = True,ax = ax[i//5,i%5],cbar=  False,xticklabels= False,yticklabels= False,vmax = 0.6,vmin = 0)

fig.tight_layout()



