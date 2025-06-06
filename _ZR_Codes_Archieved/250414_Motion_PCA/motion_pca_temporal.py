'''
This script will show how to do PCA on temporal data, making it possible for 


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
Do temporal PCA, and compare it with motion.
'''
from Signal_Functions.Pattern_Tools import Do_PCA
pcable_data = series[:,mask>0].T

PC_comps,point_coords,pca = Do_PCA(pcable_data,feature = 'Time',pcnum = 20,method = 'auto')
plt.plot(np.cumsum(pca.explained_variance_ratio_))

#%% recover graph.
rec_graph = np.zeros(mask.shape)
# rec_graph[mask>0] = PC_comps[9,:]
rec_graph[mask>0] = point_coords[:,3]
sns.heatmap(rec_graph,center = 0,square = True)



#%%
'''
Step 2, getting motion correlated graph.
This part will generate each pix's best corr with given motion curve.
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
conv_kernel = calculate_lagged_correlations(pd.Series(avr_speed), pd.Series(series.mean(-1).mean(-1)),0,14)
speed_conv = np.convolve(avr_speed,conv_kernel)[:len(avr_speed)]

# cut boulders
speed_conv = speed_conv[20:-20]
pcable_data = pcable_data[:,20:-20]
#%% calculate corr map.
corr_pix = np.zeros(len(pcable_data))
for i in tqdm(range(len(pcable_data))):
    r,p = pearsonr(speed_conv,pcable_data[i,:])
    corr_pix[i]=r
#%% show corr graph.
rec_graph = np.zeros(mask.shape)
rec_graph[mask>0] = corr_pix
# rec_graph[mask>0] = point_coords[:,0]
sns.heatmap(rec_graph,center=0.3,square = True,vmax = 0.7,vmin = -0.1,xticklabels = False,yticklabels = False,cbar = False,cmap = 'bwr')

#%% scatter of points in PC.
'''
Stats of PC comps with motion.
'''
point_num = len(corr_pix)
pc_time = pd.DataFrame(0.0,index = range(point_num),columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','Motion_Corr'])

counter = 0
for i in tqdm(range(len(corr_pix))):
    c_corr = corr_pix[i]
    pc_time.iloc[counter,10]=c_corr
    for j in range(10):
        c_pc = point_coords[i,j]
        pc_time.iloc[counter,j] = c_pc
    counter += 1
#%%
pc_time = pc_time.astype('f8')
fig,ax = plt.subplots(ncols = 1,nrows=1,dpi = 240,figsize = (6,5))
sns.scatterplot(data = pc_time,x = 'PC1',y = 'PC2',hue = 'Motion_Corr',s = 2,lw=0,palette='rainbow')

#%% Plot top 10 PCs.
fig,ax = plt.subplots(ncols = 5,nrows=2,dpi = 300,figsize = (8,4))

for i in range(10):
    rec_graph = np.zeros(mask.shape)
    rec_graph[mask>0] = point_coords[:,i]
    sns.heatmap(rec_graph,center = 0,square = True,ax = ax[i//5,i%5],cbar=  False,xticklabels= False,yticklabels= False,vmax = 40,vmin = -40)

fig.tight_layout()
#%% calculate corr.
corrs = []
ps = []
for i in range(10):
    r,p = pearsonr(speed_conv,PC_comps[i,20:-20])
    corrs.append(r)
    ps.append(p)