
'''
Here we try to add an api for nan problems. After this fix, nan value will be filled by nearby mean values be default.

'''


#%%
from OI_Functions.OIS_Preprocessing import Single_Folder_Processor
import numpy as np
import OI_Functions.Common_Functions as cf
import matplotlib.pyplot as plt
import seaborn as sns

wp=r'D:\ZR\_Data_Temp\Ois200_Data\red_speckle'
#%% Generate 
Single_Folder_Processor(wp,subfolder='Preprocessed',save_format='python')
#%% let's see the frame drop problems here.
a = np.load(r'D:\ZR\_Data_Temp\Ois200_Data\red_speckle\Preprocessed\red.npy')