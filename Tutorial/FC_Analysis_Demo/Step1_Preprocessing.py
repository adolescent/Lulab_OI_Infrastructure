'''
This part will change ois captured '.bin' file into standard format.

Function of each part will be explained in annotation.
'''


#%%
'''
Part 1, change '.bin' file into '.npy' file, This function is already packed, so you only need to run this.

'''
import Common_Functions as cf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from OI_Functions.OIS_Preprocessing import Single_Folder_Processor # this function is standard bin transformer.

datafolder = r'D:\ZR\_Data_Temp\Ois200_Data\Full_Demo'
Single_Folder_Processor(datafolder,subfolder='Preprocessed',save_format='python')

# ai file are all 12 channel logged 
# red file (or other name) are raw frame stacks captured.

#%%
'''
Part 2, bin graph into fps required. Usually bin is necessary for Blood-Oxygen level signal significance. Example data is captured in 30Hz, we usually bin it into 5Hz.
'''
savepath = r'D:\ZR\_Data_Temp\Ois200_Data\Full_Demo\Preprocessed'
raw_r_series = np.load(cf.join(savepath,'\Red.npy'))
print(raw_r_series.shape)
_,height,width = raw_r_series.shape
bin = 6
binned_num = len(raw_r_series)//bin # we bin 30 to 5, average every 6 frame
binned_r_series = raw_r_series[:binned_num*bin,:,:].reshape(binned_num,bin,height,width)
binned_r_series = binned_r_series.mean(1).astype('u2')# use uint 2 to save HD space.

avr_graph = binned_r_series.mean(0)
plt.imshow(avr_graph,cmap='gray')

np.save(cf.join(savepath,'binned_r_series'),binned_r_series)
#%%
'''
Part 3, align graph into standard space.
**This part cannot be done from remote vscode as gui unable to transfer.**
As the stack is captured in 256x256, we choose bin=4 in standard model.
'''
from Align_Tools import Match_Pattern

MP = Match_Pattern(avr = avr_graph,bin=4,lbd=4.2) 
# lbd is lambda-bregma-distance, you can physically measure the distance between them to match it more accurately.
MP.Select_Anchor()
MP.Fit_Align_Matrix()
#%%
# after model fit, transfer data points.
trans_series= MP.Transform_Series(stacks=binned_r_series)
np.save(cf.join(savepath,'Aligned_Series'),trans_series)
print(trans_series.shape)

# save avr graph for chamber mask plotting. 
# we use cv2 to make sure the graph size unchaged.
cv2.imwrite(cf.join(savepath,'Average_After.png'),trans_series.mean(0).astype('u2'))
#%%
'''
Part 4, Getting dR/R series.
This part will detrend R series, calculating dR/R series.
We also save space filted graph to flatten space info.
We save Z series together, for avoiding light diff. Clip required.

'''
# Detrend
from Signal_Functions.Filters import Signal_Filter_1D
from scipy.ndimage import gaussian_filter




# HP_graph = gaussian_filter(input = clipped_graph, sigma = HP_sigma)
# LP_graph = gaussian_filter(input = clipped_graph, sigma = LP_sigma)
# filted_graph = (LP_graph-HP_graph)


#%%
'''
Part 5, Getting area-wise data matrix, for further pattern and pair correlation analysis.
'''
# before this procedure, chamber mask is strongly recommended.


#%% Test run part

