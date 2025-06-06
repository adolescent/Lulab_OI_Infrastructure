'''
This script will try to do PCA on whole series, and find 

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
Step 1, ROC curve of speed.
'''
thres = np.linspace(0,1,101)*(avr_speed.max())
probs = np.zeros(len(thres))

for i,c_thres in enumerate(thres):
    c_prob = (avr_speed<c_thres).sum()/len(avr_speed)
    probs[i] = c_prob

plt.plot(thres,probs)

id90 = np.where(probs>0.9)[0][0]
id90_thres = thres[id90]
print(f'90% speed lower than {id90_thres:.4f}')


#%%
'''
Step 2, getting lag and align response and motion.
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
# getting lag = 8. let's lag motion.
used_lag = 8
used_series = series[used_lag:,:,:]
used_avr_speed = avr_speed[:-used_lag]


#%%
'''
Step 3, Do PCA, and getting PC comps, comparing it with speed avr.
'''
from Signal_Functions.Pattern_Tools import Do_PCA
pcable_data = series[:,mask>0].T

PC_comps,point_coords,pca = Do_PCA(pcable_data,feature = 'Area',pcnum = 20,method = 'auto')
plt.plot(np.cumsum(pca.explained_variance_ratio_))

#%% recover graph.
rec_graph = np.zeros(mask.shape)
rec_graph[mask>0] = PC_comps[9,:]
# rec_graph[mask>0] = point_coords[:,0]
sns.heatmap(rec_graph,center = 0,square = True)
#%% Plot top 10 PCs.
fig,ax = plt.subplots(ncols = 5,nrows=2,dpi = 300,figsize = (8,4))

for i in range(10):
    rec_graph = np.zeros(mask.shape)
    rec_graph[mask>0] = PC_comps[i,:]
    sns.heatmap(rec_graph,center = 0,square = True,ax = ax[i//5,i%5],cbar=  False,xticklabels= False,yticklabels= False,vmax = 0.015,vmin = -0.015)

fig.tight_layout()


#%%
'''
Step 4 compare PC weight with motion value.
'''
from scipy.stats import pearsonr,spearmanr

start = 0
end = 18000
pc=1

plt.scatter(point_coords[start:end,pc],used_avr_speed[start:end],s=2,c=used_avr_speed[start:end],alpha = 0.6)
# plt.scatter(PC_comps[0,start:end],used_avr_speed[start:end],s=2,c=used_avr_speed[start:end],alpha = 0.6)
pearsonr(point_coords[start:end,pc],used_avr_speed[start:end])


#%% NOT WORKING, DECRYPTED.
# '''
# All-value data seems ambigous. This might because of Blood—Oxygen Delay = = 
# I'll try to bin results in 1 or 2 min,making it more stable.
# '''
# binsize = 60
# fps =5
# bin_frame = binsize*fps
# winnum = len(used_avr_speed)//bin_frame

# binned_speed = used_avr_speed[:winnum*bin_frame].reshape(winnum,bin_frame).mean(-1)
# binned_pc_weights = point_coords[:winnum*bin_frame,:].reshape((winnum,bin_frame,20)).mean(1)

# # plt.plot(binned_speed)
# # plt.plot(used_avr_speed)
# plt.scatter(binned_pc_weights[:,0],binned_speed,s=2,alpha = 0.6)
#%%
'''
Let's try to convolve blood response curve to motion signal. Use lagged correlation series as kernel.
'''
conv_kernel = calculate_lagged_correlations(pd.Series(avr_speed), pd.Series(series.mean(-1).mean(-1)),0,14)
speed_conv = np.convolve(avr_speed,conv_kernel)[:len(avr_speed)]
#%% PCA on conved speed
start = 0
end = -1
pc=0

cpc_weight = point_coords[start:end,pc]
c_speed = speed_conv[start:end]
# filt if needed
# cpc_weight = Signal_Filter_1D(cpc_weight,False,0.5,5)
# c_speed = Signal_Filter_1D(c_speed,False,0.5,5)

plt.scatter(cpc_weight,c_speed,s=2,c=c_speed,alpha = 0.6)
pearsonr(cpc_weight,c_speed)

#%% 
# Plot correlation of each PC's weight and motion
motion_corr = []
motion_p = []
for i in range(10):
    c_coord = point_coords[start:end,i]
    c_speed = speed_conv[start:end]
    r,p = pearsonr(c_coord,c_speed)
    motion_corr.append(r)
    motion_p.append(p)
    

#%% Getting ROC curve of conved speed, and getting 10% and 90% thres.
    
import matplotlib.lines as mlines
# cut head and tail.
speed_conv = speed_conv[20:-20] 
point_coords = point_coords[20:-20,:]

thres = np.linspace(0,1,1001)*(speed_conv.max())
probs = np.zeros(len(thres))

for i,c_thres in enumerate(thres):
    c_prob = (speed_conv<c_thres).sum()/len(speed_conv)
    probs[i] = c_prob



id90 = np.where(probs>0.9)[0][0]
id90_thres = thres[id90]
id10 = np.where(probs<0.1)[0][-1]
id10_thres = thres[id10]
print(f'90% speed lower than {id90_thres:.4f}')
print(f'10% speed lower than {id10_thres:.4f}')


fig,ax = plt.subplots(ncols = 1,nrows= 1,dpi = 240,figsize = (5,4))
l1 = mlines.Line2D([id10_thres, id10_thres], [0,0.1], color='red', linestyle='dashed', linewidth=1)
l11 = mlines.Line2D([0, id10_thres], [0.1,0.1], color='red', linestyle='dashed', linewidth=1)
l2 = mlines.Line2D([id90_thres, id90_thres], [0,0.9], color='blue', linestyle='dashed', linewidth=1)
l22 = mlines.Line2D([0, id90_thres], [0.9,0.9], color='blue', linestyle='dashed', linewidth=1)
ax.plot(thres,probs)
ax.add_line(l1)
ax.add_line(l11)
ax.add_line(l2)
ax.add_line(l22)
ax.set_ylim(0,1)
ax.set_xlim(0,40)
#%% getting motion max and min lists.
low_id = np.where(speed_conv<id10_thres)[0]
high_id = np.where(speed_conv>id90_thres)[0]
frame_num = len(low_id)+len(high_id)
weights = pd.DataFrame(index = range(frame_num*10),columns = ['Motion','PC','PC_Weights','Speed_conv'])

counter = 0
for i,c_id in tqdm(enumerate(low_id)):
    c_pcs = point_coords[c_id,:]
    c_motion = speed_conv[c_id]
    for j in range(10):
        weights.loc[counter,:] = ['Low',j+1,c_pcs[j],c_motion]
        counter+=1

for i,c_id in tqdm(enumerate(high_id)):
    c_pcs = point_coords[c_id,:]
    c_motion = speed_conv[c_id]
    for j in range(10):
        weights.loc[counter,:] = ['High',j+1,c_pcs[j],c_motion]
        counter+=1
weights['PC_Weights'] = weights['PC_Weights'].astype('f8')
#%% plot PC weights between low and high.
        
fig,ax = plt.subplots(ncols = 1,nrows= 1,dpi = 240,figsize = (8,5))
# sns.boxplot(data = weights,x = 'PC',y = 'PC_Weights',hue = 'Motion',ax = ax,showfliers = False)
# sns.violinplot(data = weights,x = 'PC',y = 'PC_Weights',hue = 'Motion',ax = ax)

sns.barplot(data = weights,x = 'PC',y = 'PC_Weights',hue = 'Motion',ax = ax)

ax.set_ylim(-70,70)

#%% 
'''
PCA-TTEST, return a motion-negative ttest map. This map shows PC's anti-motion network.
average this network and show it on graph.

'''
from scipy.stats import ttest_ind
pc_diff = np.zeros(10)

for i in range(1,11):
    c_pc = weights.groupby('PC').get_group(i)
    c_high = c_pc.groupby('Motion').get_group('High')['PC_Weights']
    c_low = c_pc.groupby('Motion').get_group('Low')['PC_Weights']
    r,p = ttest_ind(c_high,c_low)
    # r = c_high.mean()-c_low.mean()
    pc_diff[i-1]=r

# add up PC diff on pc_comps
motion_comp = np.dot(pc_diff[:],PC_comps[:10,:])

# recover motion comp
rec_graph = np.ones(mask.shape)*0.3
rec_graph[mask>0] = motion_comp
# rec_graph[mask>0] = point_coords[:,0]
sns.heatmap(rec_graph,center=0.3,square = True,vmax = 0.7,vmin = -0.1,xticklabels = False,yticklabels = False,cbar = False,cmap = 'bwr')
#%% get motion high and motion low subgraph.
raw_series_1d = pcable_data[:,20:-20]
low_raw = raw_series_1d[:,low_id]
high_raw = raw_series_1d[:,high_id]

t_graph = np.zeros(shape = len(low_raw))
p_graph = np.zeros(shape = len(low_raw))
for i in tqdm(range(len(low_raw))):
    pix_high = high_raw[i,:]
    pix_low = low_raw[i,:]
    t_graph[i],p_graph[i] = ttest_ind(pix_high,pix_low,equal_var=False)
#%% plot t graph high-low
rec_graph = np.zeros(mask.shape)
rec_graph[mask>0] = t_graph
sns.heatmap(rec_graph,center = 0,vmax = 10,vmin = -10,square = True,xticklabels = False,yticklabels = False,cmap = 'bwr')