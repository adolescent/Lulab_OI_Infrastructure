
#%%

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import copy
import Common_Functions as cf
from OI_Functions.Align_Tools import Match_Pattern
from Brain_Atlas.Atlas_Mask import Mask_Generator
from OI_Functions.OIS_Preprocessing import Single_Folder_Processor
#%%
wp=r'D:\ZR\_Data_Temp\Ois200_Data\Affine_Data\Cutted'
Single_Folder_Processor(wp,subfolder='Preprocessed',save_format='python')

#%% Load R value matrix
r_series = np.load(r'D:\ZR\_Data_Temp\Ois200_Data\Affine_Data\Cutted\Preprocessed\Red.npy')
avr = r_series.mean(0)
#%% Align matrix 
# try this till graph is okay to you.
MP = Match_Pattern(avr,bin=4)
MP.Select_Anchor()
MP.Fit_Align_Matrix()

#%% If done, generate transformed matrix.
stacks = MP.Transform_Series(r_series)
np.save(cf.join(r'D:\ZR\_Data_Temp\Ois200_Data\Affine_Data\Cutted\Preprocessed','Aligned_Stacks.npy'),stacks)
affine_avr = stacks.mean(0)
#%% get single brain area's response curve.
min_pix = 100
chamber_mask = np.ones(shape=(MP.height,MP.width))
MG = Mask_Generator(bin=4)
all_area_name = MG.all_areas # look up required brain area.

v1_l_mask = MG.Get_Mask(area ='SSp-tr',LR = 'L')*chamber_mask
if v1_l_mask.sum()<min_pix:
    raise ValueError('Mask out of chamber.')

v1_l = affine_avr*v1_l_mask
plt.imshow(v1_l,cmap='gray')

#%% get v1_l series
v1_l_r_series = stacks[:,v1_l>0]
plt.plot(v1_l_r_series.mean(1))