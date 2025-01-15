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
from tqdm import tqdm
import pandas as pd
from Brain_Atlas.Atlas_Mask import Mask_Generator

from OI_Functions.OIS_Preprocessing import Single_Folder_Processor # this function is standard bin transformer.
#%%
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
bin_time = 6
binned_num = len(raw_r_series)//bin_time # we bin 30 to 5, average every 6 frame
binned_r_series = raw_r_series[:binned_num*bin_time,:,:].reshape(binned_num,bin_time,height,width)
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
binned_r_series = np.load(r'D:\ZR\_Data_Temp\Ois200_Data\Full_Demo\Preprocessed\binned_r_series.npy')
avr_graph = binned_r_series.mean(0)
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
Part 4, Getting dR/R series and Z series.
This part will detrend R series, calculating dR/R series.
We also save space filted graph to flatten space info.
We save Z series together, for avoiding light diff. Clip required.

'''
from Signal_Functions.Filters import Signal_Filter_1D
from scipy.ndimage import gaussian_filter
wp = r'D:\ZR\_Data_Temp\Ois200_Data\Full_Demo\Preprocessed'
raw_r_series = np.load(cf.join(wp,r'Aligned_Series.npy'))
mask = cv2.imread(cf.join(wp,'Chamber_mask.png'),0)>0
HP = 0.005
LP = 1
fps = 5


img_num,height,width = raw_r_series.shape
# Detrend graph, and do spacial filter if required.
filt_r_series = np.zeros(shape = (raw_r_series.shape),dtype = 'f8')
for i in tqdm(range(height)):
    for j in range(width):
        c_series = raw_r_series[:,i,j]
        if c_series.sum() !=0:
            c_filt_series = Signal_Filter_1D(c_series,HP,LP,5)
            filt_r_series[:,i,j] = c_filt_series

# generate drr and z series.
drr_series = np.nan_to_num((filt_r_series-filt_r_series.mean(0))/filt_r_series.mean(0))
z_series = np.nan_to_num(drr_series/drr_series.std(0))
# save raw data series, for further analysis, you will need clip or normalize.
np.save(cf.join(wp,'drr_series'),drr_series)
np.save(cf.join(wp,'z_series'),z_series)

# as you can see, effect of bv can be supressed easily by z value.
fig,ax = plt.subplots(ncols=2,nrows=1,dpi = 200,figsize = (8,5))
sns.heatmap(z_series[699,:,:],center = 0,square = True,vmax = 0.5,vmin = -0.5,ax = ax[0])
sns.heatmap(drr_series[699,:,:],center = 0,square = True,vmax = 0.005,vmin = -0.005,ax = ax[1])
# and show the distribution of z value.
plt.hist(z_series.flatten(),bins = 50)
#%% let's see the video of drr and z action series.

# I prefer Z value for further analysis.This might ignore big BV event, so be careful if you want BV signal.
import skvideo.io

# this function set 0 as 127, pack graph into a video.
def Concat_Video_Plotter(series_A,series_B,savepath,filename,fps=25):
    norm_series_A = series_A/max(series_A.max(),abs(series_A.min()))
    plotable_A = (norm_series_A*127+127).astype('u1')
    norm_series_B = series_B/max(series_B.max(),abs(series_B.min()))
    plotable_B = (norm_series_B*127+127).astype('u1')
    concat_series = np.concatenate((plotable_A,plotable_B),axis=2)

    fullpath = cf.join(savepath,filename)+'.avi'
    _,height,width = concat_series.shape
    
    writer = skvideo.io.FFmpegWriter(fullpath, outputdict={
        '-vcodec': 'rawvideo',  #use the h.264 codec
        #  '-vcodec': 'libx264',
        '-crf': '0',           #set the constant rate factor to 0, which is lossless
        '-preset':'veryslow',   #the slower the better compression, in princple, 
        # '-r':str(fps), # this only control the output frame rate.
        '-pix_fmt': 'yuv420p',
        '-vf': "setpts=PTS*{},fps={}".format(25/fps,fps) ,
        '-s':'{}x{}'.format(width,height)
    }) 

    for frame in tqdm(concat_series):
        # cv2.imshow('display',frame)
        writer.writeFrame(frame)  #write the frame as RGB not BGR
        # time.sleep(1/fps)

    writer.close() #close the writer
    cv2.destroyAllWindows()
    return True

# input_series = np.clip(z_series,-5,5)
Concat_Video_Plotter(np.clip(drr_series[3000:6000]*mask,-0.005,0.005),np.clip(z_series[3000:6000]*mask,-1,1),wp,'Compare',10) # 2x speed.
# Video_Plotter(np.clip(z_series[3000:6000]*mask,-1,1),wp,'z',10) # 2x speed.

############### DECREPTED ############
# #%% Next part will try to do spatial filter for brightness
# #%% This part seems not to be necessary, z score can fix all the problem.
# HP_sigma = 100
# LP_sigma = 0.75

# HP_graph = gaussian_filter(input = avr_graph , sigma = HP_sigma,mode='constant')
# LP_graph = gaussian_filter(input = avr_graph , sigma = LP_sigma,mode='constant')
# filted_graph = (LP_graph-HP_graph)
# plt.imshow(filted_graph*mask,cmap = 'gray')
# # plt.imshow(HP_graph,cmap = 'gray')

#%%
'''
Part 5, Getting area-wise data matrix, for further pattern and pair correlation analysis.
'''
# before this procedure, chamber mask is strongly recommended.

wp = r'D:\ZR\_Data_Temp\Ois200_Data\Full_Demo\Preprocessed'
mask = cv2.imread(cf.join(wp,'Chamber_mask.png'),0)>0


z_series = np.load(r'D:\ZR\_Data_Temp\Ois200_Data\Full_Demo\Preprocessed\z_series.npy')
series = np.clip(z_series,-3,3)*mask


from Atlas_Corr_Tools import Atlas_Data_Tools,Contra_Similar

ADT = Atlas_Data_Tools(series=series,bin=4,min_pix=30)
ADT.Get_All_Area_Response()
Area_Response = ADT.Area_Response
Area_Response_Heatmap = ADT.Combine_Response_Heatmap()
Corr_Matrix = ADT.Get_Corr_Matrix(win_size=1500,win_step = 300,keep_unilateral=False)
sns.heatmap(Corr_Matrix[1],center=  0.5,vmax = 1,square = True)
# cf.Save_Variable(wp,'Atlas_Infos',ADT)


#%% It's also possible for contralateral consistency calculation.
# similary, you will need to provide win_len and win_step.
winsize = 1500
winstep = 300
winnum = (len(series)-winsize)//winstep+1
contra_sims = np.zeros(shape = (winnum,330,285),dtype='f8')
for i in tqdm(range(winnum)):
    c_slide = series[winstep*i:winstep*i+winsize,:,:]
    contra_sims[i,:,:] = Contra_Similar(c_slide,bin=4)

np.save(cf.join(wp,'Contralateral_Similar'),contra_sims)

#%% Plot graphs.
savepath = cf.join(wp,'Contralateral')
cf.mkdir(savepath)
for i in tqdm(range(len(contra_sims))):
    c_corr = contra_sims[i,:,:]
    plt.clf()
    # plt.cla()
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize = (5,5),dpi = 240)
    sns.heatmap(c_corr,xticklabels=False,yticklabels=False,square=True,ax = ax,center=0.8,vmax = 1,vmin =0.6)
    fig.savefig(cf.join(savepath,f'{10000+i}.png'))
    plt.close(fig)

