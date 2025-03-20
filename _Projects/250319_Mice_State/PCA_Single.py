'''
This script will generate single-case PCA. by cutting frames into small pieces and concat them together.
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
from scipy.stats import pearsonr

# load frame
# wp = r'D:\ZR\_Data_Temp\Ois200_Data\activity_analysis\47#'
wp = r'D:\ZR\_Data_Temp\Ois200_Data\activity_analysis\A6'
series = np.load(cf.join(wp,'z_series.npy'))
chamber_mask = cv2.imread(cf.join(wp,'chambermask.png'),0)>0
series = series*chamber_mask
series = np.clip(series,-5,5)

avr_speed = np.load(cf.join(wp,'whisker_speed_5Hz.npy'))
#%% get lagged corr.
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

from scipy.signal import find_peaks,peak_widths
cutted_speed = avr_speed[:]

peaks, properties = find_peaks(cutted_speed, prominence=2,distance=10,height=1) 

plt.plot(cutted_speed)
plt.plot(peaks,cutted_speed[peaks], "x")
plt.show()

#%% get peak infos
def get_surrounding_indices(all_len,peaks,half_width,lag):
    # Remove duplicate peaks and convert to set for quick look-up

    for i,c_peak in enumerate(peaks):
        start = max(0,c_peak-half_width+lag)
        end = min(all_len,c_peak+half_width+lag)
        index = np.arange(start,end)
        if i == 0:
            all_index = copy.deepcopy(index)
        else:
            all_index = np.concatenate((all_index,index))
    peak_index = np.unique(all_index)
    # get index not in peak
    no_peak_index = []
    for i in range(all_len):
        if i not in peak_index:
            no_peak_index.append(i)
    no_peak_index = np.array(no_peak_index)

    return peak_index,no_peak_index
        
bold_lag = 4
peak_halfwidth = 15
peak_index,no_peak_index = get_surrounding_indices(len(cutted_speed),peaks,peak_halfwidth,bold_lag)
#%% save peak and no-peak response for PCA
resting_series = series[no_peak_index,:,:]
resting_series_1d = resting_series[:,chamber_mask].T
# del resting_series
#%%
from Signal_Functions.Pattern_Tools import Do_PCA
PC_Comp_raw,coords_raw,model_raw = Do_PCA(resting_series_1d,feature='Area',pcnum=75)
print(f'PC_Comp are in shape N_Comp*N_feature:{PC_Comp_raw.shape}')
print(f'Coords_raw are in shape N_Sample*N_Comp:{coords_raw.shape}')
#%% VAR description
plt.plot(np.cumsum(model_raw.explained_variance_ratio_))

print(f'First PC explained VAR {model_raw.explained_variance_ratio_[0]*100:.2f}%')
print(f'Top 20 PC explained var: {np.sum(model_raw.explained_variance_ratio_[:20]*100):.2f}%')

#%% recover pc

def Recover_Map(vec,mask):
    recover = copy.deepcopy(mask).astype('f8')
    recover[recover==True] = vec
    return recover


fig,ax = plt.subplots(ncols=1,nrows=1,figsize = (4,4),dpi = 180)
sns.heatmap(Recover_Map(PC_Comp_raw[0,:],chamber_mask),center = 0,cbar=False,ax = ax,xticklabels=False,yticklabels=False,square=True)
#%% save PC's pattern.
savepath = cf.join(wp,'PC_Spatial')
cf.mkdir(savepath)
for i in tqdm(range(len(PC_Comp_raw))):
    c_map = Recover_Map(PC_Comp_raw[i,:],chamber_mask)

    plt.clf()
    # plt.cla()
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize = (5,5),dpi = 240)
    sns.heatmap(c_map,xticklabels=False,yticklabels=False,square=True,ax = ax,center=0)
    fig.savefig(cf.join(savepath,f'{10000+i}.png'))
    plt.close(fig)


#%%############
'''
Making Hierarchical Clustering.
For calculation convenience, we have to do dim reduction.
'''

series_reshape = resting_series.reshape(len(resting_series),66,5,57,5)
# we need mask in boulder adjusted, so I cannot avr graph directly.
series_bin = np.zeros(shape = (len(resting_series),66,57),dtype='f8')
mask_reshape = (resting_series.std(0)>0).reshape(66,5,57,5)
for i in tqdm(range(66)):
    for j in range(57):
        c_block = series_reshape[:,i,:,j,:].sum(-1).sum(-1)
        c_masknum = mask_reshape[i,:,j,:].sum()
        if c_masknum>0: # have a value.
            series_bin[:,i,j] = c_block/c_masknum

print(series_bin.shape)

# as you can see, the mask have different values in boulder.
fig,ax = plt.subplots(ncols=2,nrows=1,figsize = (5,3),dpi = 180)
ax[0].imshow(series_bin[5500,:,:])
ax[1].imshow(mask_reshape.mean(-1).mean(-2))

# delete var to save memory.
del resting_series,series_reshape
#%% 1d for binned graph.
on_mask = series_bin.std(0)>0
on_mask *= on_mask[:,::-1]
on_response = series_bin[:,on_mask==True].T
print(f'Mask in shape {on_mask.shape}, total ON pix :{on_mask.sum()}')
print(f'Graph have already been vectorized: {on_response.shape}')

#%%

from scipy.spatial.distance import pdist,squareform

distance_metric = 'euclidean'
# cityblock return L1 norm of data. Different matrix return some different results.
# Multiple options: 'euclidean', 'correlation', 'cityblock','chebyshev',.....

distance_matrix = pdist(on_response, metric=distance_metric)
distance_matrix.shape
#%%
square_matrix = squareform(distance_matrix) 
sns.heatmap(square_matrix)
# you can also use this function to transform distance matrix into dense 1D corr info.
recovered_matrix = squareform(square_matrix)
# test whether recovered matrix is the same as original
print((recovered_matrix == distance_matrix).min())

#%% generating linkage matrix.
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
Z = linkage(distance_matrix, method='ward',metric=distance_metric )
print(Z.shape)
print(Z[0])
fig,ax = plt.subplots(ncols=1,nrows=1,figsize = (4,6),dpi = 180)
a = dendrogram(
    Z,
    truncate_mode='lastp',  # Show only the last p merged clusters
    p=25000,                   #Steps of cluster to show. set very big will show whole dentrogram.
    show_leaf_counts=True,
    leaf_rotation=90, # rotation angle of leaf 
    leaf_font_size=6, # fontsize of leaf
    ax = ax,
    orientation= 'top' # Orientation of cluster going on. top as default.
)
ax.set_title('Dendrogram')
ax.set_xlabel('Samples')
ax.set_ylabel('Distance')
ax.set_xticks([]) # Mute x label if you want full graph.
#%%
# method 1, show cluster by selecting cluster num directly.
n_clusters = 5
clusters = fcluster(Z, t=n_clusters, criterion='maxclust')
print(clusters) # cluster is a list of id, indicating the cluster of each pix.

# then visualize.
recover = copy.deepcopy(on_mask).astype('i4')
recover[recover==True] = clusters
plt.imshow(recover,cmap='jet')