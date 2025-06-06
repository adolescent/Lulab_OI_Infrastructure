'''
This script shows max corr on different frequency band.

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
from Signal_Functions.Spectrum_Tools import *


all_path = cf.Get_Subfolders(r'D:\_DataTemp\OIS\All_WT_Datasets')
save_path = r'D:\_DataTemp\OIS\Motion_Midvars'


#%%
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

for i,cloc in enumerate(all_path):
    c_full = np.load(cf.join(cloc,'Z_Series_Full.npy')).mean(-1)
    c_high = np.load(cf.join(cloc,'Z_Series_High.npy')).mean(-1)
    c_medium = np.load(cf.join(cloc,'Z_Series_Medium.npy')).mean(-1)
    c_low = np.load(cf.join(cloc,'Z_Series_Low.npy')).mean(-1)
    c_whisker = np.load(cf.join(cloc,'whisker_speed_5Hz.npy'))
    # we found that correlation consists in freq lower than 0.2Hz.