#%%

# import and basic path part
import OI_Functions.Common_Functions as cf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm
import pandas as pd
from Brain_Atlas.Atlas_Mask import Mask_Generator
from OI_Functions.OIS_Preprocessing import Single_Folder_Processor
import re


data_folder = r'/mnt/6551E1FB6F594FFC/Test_Folder/Short_Demo'

Single_Folder_Processor(data_folder,subfolder='Preprocessed',save_format='python',keepna=False)
# keepna parameter will maintain lost frame as 0, otherwise these frames are filled with previous frame.

# then we load the pre-processed frame.
wp = cf.join(data_folder,r'Preprocessed')
raw_r_series = np.load(cf.join(wp,'Red.npy'))
print(raw_r_series.shape)


#%%
if __name__ == '__main__':
    test_path = r'D:\ZR\_Data_Temp\Ois200_Data\240417_M3_Bars\test_exposetime0.1ms_Vbar_Run01\Preprocessed'
    all_frames = np.load(cf.join(test_path,'Red.npy'))
    stim_ids = np.load(cf.join(test_path,'ai_series.npy'))
    # and read stim txt here.
    txt_path = r'D:\ZR\_Data_Temp\Ois200_Data\240417_M3_Bars\stimid_folder_0240419\Run01_V\stim_id_17_14_44.txt'
    txt_file = open(txt_path, "r")
    stim_data = txt_file.read()
    stim_series = re.split('\n|,',stim_data)[:25]

#%% try to fix the dark stripes.
all_frames_12bit = (all_frames//16)
plt.imshow(all_frames_12bit.mean(0))

