'''
This script will stat motions, and return estimated motion threshold, for 10% and 90%, getting it's 

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

for i,cloc in enumerate(all_path):
    c_motion = np.load(cf.join(cloc,'whisker_speed_5Hz.npy'))
    
