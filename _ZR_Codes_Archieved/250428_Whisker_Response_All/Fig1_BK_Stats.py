'''
This is backgroud codes for Figure 1, getting slide correlation between 

'''



#%% import and path cycle.
'''
First part, preprocessing data to get Z series and data on different power bands.


'''
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

#%% get power of 3 band, generate filted and clipped z series.
clip_value = 5
fps = 5
band_full = (0.005,2)
band_a = (0.005,0.2)
band_b = (0.2,0.5)
band_c = (0.5,2)

for i,cloc in enumerate(all_path):
    series_raw = np.load(cf.join(cloc,'Transformed_R_Series.npy'))

    # bin mask and remover center 10 pix.
    c_mask = cv2.imread(cf.join(cloc,'Mask.png'),0)>0
    c_mask[:,138:149]=0
    value_mask = series_raw.std(0)>0
    c_mask = c_mask*value_mask
    c_mask = c_mask*c_mask[:,::-1]
    # plt.imshow(c_mask)
    np.save(cf.join(cloc,'Mask'),c_mask)

    # filt 3 band power
    series_raw = series_raw[:,c_mask==True]
    img_num,pix_num = series_raw.shape
    filt_r_series_full = np.zeros(shape = (series_raw.shape),dtype = 'f8')
    filt_r_series_low = np.zeros(shape = (series_raw.shape),dtype = 'f8')
    filt_r_series_medium = np.zeros(shape = (series_raw.shape),dtype = 'f8')
    filt_r_series_high = np.zeros(shape = (series_raw.shape),dtype = 'f8')

    for j in tqdm(range(pix_num)):# actually there are quicker ways, but I'm lazy.
        c_series = series_raw[:,j]
        c_filt_series_full = Signal_Filter_1D(c_series,HP_freq=band_full[0],LP_freq=band_full[1],fps=fps)# you can open this function to see what these parameters means.
        c_filt_series_low = Signal_Filter_1D(c_series,HP_freq=band_a[0],LP_freq=band_a[1],fps=fps)# you can open this function to see what these parameters means.
        c_filt_series_medium = Signal_Filter_1D(c_series,HP_freq=band_b[0],LP_freq=band_b[1],fps=fps)# you can open this function to see what these parameters means.
        c_filt_series_high = Signal_Filter_1D(c_series,HP_freq=band_c[0],LP_freq=band_c[1],fps=fps)# you can open this function to see what these parameters means.
        filt_r_series_full[:,j] = c_filt_series_full
        filt_r_series_low[:,j] = c_filt_series_low
        filt_r_series_medium[:,j] = c_filt_series_medium
        filt_r_series_high[:,j] = c_filt_series_high

    # get drr and z series of given series, for each band.
    drr_series_full = np.nan_to_num((filt_r_series_full-filt_r_series_full.mean(0))/filt_r_series_full.mean(0))
    drr_series_low = np.nan_to_num((filt_r_series_low-filt_r_series_low.mean(0))/filt_r_series_low.mean(0))
    drr_series_medium = np.nan_to_num((filt_r_series_medium-filt_r_series_medium.mean(0))/filt_r_series_medium.mean(0))
    drr_series_high = np.nan_to_num((filt_r_series_high-filt_r_series_high.mean(0))/filt_r_series_high.mean(0))
    # Z value supress blood-vessle pixel with high std.
    z_series_full = np.nan_to_num(drr_series_full/drr_series_full.std(0))
    z_series_full = np.clip(z_series_full,-clip_value,clip_value)
    z_series_low = np.nan_to_num(drr_series_low/drr_series_low.std(0))
    z_series_low = np.clip(z_series_low,-clip_value,clip_value)
    z_series_medium = np.nan_to_num(drr_series_medium/drr_series_medium.std(0))
    z_series_medium = np.clip(z_series_medium,-clip_value,clip_value)
    z_series_high = np.nan_to_num(drr_series_high/drr_series_high.std(0))
    z_series_high = np.clip(z_series_high,-clip_value,clip_value)
    np.save(cf.join(cloc,'Z_Series_Full'),z_series_full)
    np.save(cf.join(cloc,'Z_Series_Low'),z_series_low)
    np.save(cf.join(cloc,'Z_Series_Medium'),z_series_medium)
    np.save(cf.join(cloc,'Z_Series_High'),z_series_high)
    

