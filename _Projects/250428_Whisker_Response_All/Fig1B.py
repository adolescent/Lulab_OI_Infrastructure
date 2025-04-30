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

all_path = cf.Get_Subfolders(r'D:\ZR\_Data_Temp\Ois200_Data\activity_analysis\All_WT_Datasets')

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
# Fig 1C, correlate bv response with whisker motion, getting the max lag and decay plot

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

for i,cloc in enumerate(all_path):
    pass
