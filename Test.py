#%%

import OI_Functions.Common_Functions as cf
import OI_Functions.OIS_Tools as Ois_Tools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re






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

