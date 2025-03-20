'''
This script will try to estimate state from whisker motion.

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
wp = r'D:\ZR\_Data_Temp\Ois200_Data\activity_analysis\47#'
# wp = r'D:\ZR\_Data_Temp\Ois200_Data\activity_analysis\A6'
series = np.load(cf.join(wp,'z_series.npy'))
chamber_mask = cv2.imread(cf.join(wp,'chambermask.png'),0)>0
series = series*chamber_mask
series = np.clip(series,-5,5)

#%% load whisker
import h5py    

# filename =cf.join(wp,'47#250115_FacemapPose.h5')
filename = r'D:\ZR\_Data_Temp\Ois200_Data\activity_analysis\A6\facemap\Run01-A6_FacemapPose.h5'
f = h5py.File(filename, 'r+')
dset = f['Facemap']
keys = list(dset.keys())
print(keys)
w1x = np.array(dset['whisker(I)']['x'])
w1y = np.array(dset['whisker(I)']['y'])
w2x = np.array(dset['whisker(II)']['x'])
w2y = np.array(dset['whisker(II)']['y'])
w3x = np.array(dset['whisker(III)']['x'])
w3y = np.array(dset['whisker(III)']['y'])
f.close()

#%% 
'''
Part 1, calculate whisker moving speed by dx and dy
'''
def speed_calculator(x_series,y_series):

    x_speed = np.gradient(x_series)
    y_speed = np.gradient(y_series)
    speed = np.sqrt(x_speed**2+y_speed**2)

    return speed

w1_speed = speed_calculator(w1x,w1y)
w2_speed = speed_calculator(w2x,w2y)
w3_speed = speed_calculator(w3x,w2y)

avr_speed = (w1_speed+w2_speed+w3_speed)/3
# start and end frame comes from video!
# start_frame = 48
# end_frame = 72123
start_frame = 53
end_frame = -1
avr_speed = avr_speed[start_frame:end_frame]


#%% resample motion data to match capture frequency.
from Signal_Functions.Filters import Signal_Filter_1D

from scipy.interpolate import interp1d

# Assuming series_A and series_B are NumPy arrays
time_A = np.linspace(0, 1, series.shape[0])
time_B = np.linspace(0, 1, len(avr_speed))

avr_speed_d = interp1d(time_B, avr_speed, kind='nearest')(time_A)
# speed captured in 5 Hz.
np.save(cf.join(wp,'whisker_speed_5Hz'),avr_speed_d)

#%% 
'''
First, let's compare the response and avr Z.
'''
MG = Mask_Generator(bin=4)
SS_l = MG.Get_Func_Mask(area='MO',LR='L')*chamber_mask
SS_resp = series[:,SS_l].mean(-1)
# global_avr = series.mean(-1).mean(-1)
# SS_resp = Signal_Filter_1D(SS_resp,HP_freq = 0.05,LP_freq=False,fps = 5)
#%%
fig,ax = plt.subplots(ncols=1,nrows=1,figsize = (6,4),dpi=240)
ax.plot(avr_speed_d[10000:10300])
ax.plot(SS_resp[10000:10300]*2)
# plt.plot(global_avr[10000:10300]*4)
#%% calculate autocorrelation between motion and SS signal.

def calculate_lagged_correlations(series_a, series_b, min_lag=-20, max_lag=20):
    """
    Calculate lagged correlations between two pandas Series for given lag range.
    
    Parameters:
    - series_a: pandas Series
    - series_b: pandas Series
    - min_lag: int, minimum lag (default: -20)
    - max_lag: int, maximum lag (default: 20)
    
    Returns:
    - Dict with lags as keys and correlation coefficients as values.
    """
    correlations = []
    for lag in range(min_lag, max_lag + 1):
        # Shift series_b by the current lag (negative sign to align with standard definition)
        shifted_b = series_b.shift(-lag)
        # Compute Pearson correlation, ignoring NaN values
        corr = series_a.corr(shifted_b)
        correlations.append(corr)
    correlations = np.array(correlations)
    return correlations

# Example usage:
# Assuming series_a and series_b are your pandas Series
lagged_corrs = calculate_lagged_correlations(pd.Series(avr_speed_d), pd.Series(SS_resp))
fig,ax = plt.subplots(ncols=1,nrows=1,figsize = (4,4),dpi=240)
ax.plot(lagged_corrs)
ax.axvline(x = lagged_corrs.argmax(),linestyle='--',color='gray')

ax.set_xticks(np.arange(0,40,4))
ax.set_xticklabels(np.arange(-20,20,4))

#%%
'''
Second, cut time window to 5min parts, and add up each window's total action.
'''
winsize = 300
winnum = len(avr_speed_d)//winsize
motion_value = np.zeros(winnum)

for i in range(winnum):
    c_motion = avr_speed_d[i*winsize:(i+1)*winsize]
    motion_value[i] = c_motion.sum()
    
sorted_value = copy.deepcopy(motion_value)
sorted_value.sort()
# get the most active 
plt.plot(sorted_value)
#%%
sorted_indices = np.argsort(motion_value)
n = len(motion_value)
part_size = n // 3  # Integer division for equal parts

# Split into three parts
lower_indices = sorted_indices[:part_size].tolist()
medium_indices = sorted_indices[part_size : 2 * part_size].tolist()
higher_indices = sorted_indices[2 * part_size :].tolist()

#%% calculate corr matrix


def Corr_Core(Mask_A,Mask_B,c_series):

    resp_a = c_series[:,Mask_A].mean(-1)
    resp_b = c_series[:,Mask_B].mean(-1)
    corr,_ = pearsonr(resp_a,resp_b)

    return corr

### codes below might be ugly = =
## define basic structure
corr_info_same_hemi = pd.DataFrame(columns = ['Case_Name','Case_Type','Winnum','Motion_Level','Motion_Sum','Area_A','Area_B','A_Hemi','Corr'])
corr_info_contra_hemi = pd.DataFrame(columns = ['Case_Name','Case_Type','Winnum','Motion_Level','Motion_Sum','Area_A','Area_B','A_Hemi','Corr'])

### parameters
motion_list = ['Low','Medium','High']
motion_index = [lower_indices,medium_indices,higher_indices]
hemi = ['L','R']
corr_pair = [('VI','SS'),('VI','MO'),('VI','RSP'),('SS','MO'),('SS','RSP'),('MO','RSP')]
case_name = 'A6'
case_type = 'GGC11'
framenum = len(avr_speed_d)
# bold_lag = 7
bold_lag = 5

## fill the matrix
for l,c_motion in enumerate(motion_list):
    c_motion_indices = motion_index[l]

    for i,cwin in tqdm(enumerate(c_motion_indices)):
        c_start = cwin*winsize+bold_lag
        c_end = (cwin+1)*winsize+bold_lag
        if c_end > framenum:
            # skip last window
            continue
        c_series = series[c_start:c_end,:,:]
        for j,c_hemi in enumerate(hemi):
            if c_hemi =='L':
                contra_hemi = 'R'
            else:
                contra_hemi = 'L'

            for k,c_pair in enumerate(corr_pair):
                # masks
                mask_a = MG.Get_Func_Mask(c_pair[0],c_hemi)*chamber_mask
                mask_b = MG.Get_Func_Mask(c_pair[1],c_hemi)*chamber_mask
                mask_b_contra = MG.Get_Func_Mask(c_pair[1],contra_hemi)*chamber_mask
                # corrs
                c_corr = Corr_Core(mask_a,mask_b,c_series)
                c_corr_contra = Corr_Core(mask_a,mask_b_contra,c_series)
                # fill
                corr_info_same_hemi.loc[len(corr_info_same_hemi)] = [case_name,case_type,cwin,c_motion,motion_value[cwin],c_pair[0],c_pair[1],c_hemi,c_corr]
                corr_info_contra_hemi.loc[len(corr_info_same_hemi)] = [case_name,case_type,cwin,c_motion,motion_value[cwin],c_pair[0],c_pair[1],c_hemi,c_corr_contra]

cf.Save_Variable(wp,'Corr_by_Motion_Same',corr_info_same_hemi)
cf.Save_Variable(wp,'Corr_by_Motion_Contra',corr_info_contra_hemi)

    
#%% let's test whether it is okay...
# use visual and somatosensory as model.
example_frame = copy.deepcopy(corr_info_same_hemi.groupby(['Area_A','Area_B']).get_group(('VI','SS')))
example_frame['Corr'] = example_frame['Corr'].astype('f8')

fig,ax = plt.subplots(ncols=1,nrows=1,figsize = (5,4),dpi=240)
sns.boxplot(data=example_frame,x = 'Motion_Level',y='Corr',hue = 'A_Hemi',ax = ax,width=0.5,showfliers=False)
ax.set_ylim(0.85,1)


