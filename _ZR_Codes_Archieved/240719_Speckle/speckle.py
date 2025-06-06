'''
Try to calculate some spkecle.
'''
#%% Imports
import OI_Functions.Common_Functions as cf
from OI_Functions.Stim_Frame_Align import Stim_Camera_Align
from OI_Functions.Stimulus_dRR_Calculator import dRR_Generator
from OI_Functions.Map_Subtractor import Sub_Map_Generator
import numpy as np
import re
import matplotlib.pyplot as plt
from P1_OIS_Preprocessing import One_Key_OIS_Preprocessor
import seaborn as sns
from tqdm import tqdm

raw_folder = r'D:\ZR\_Data_Temp\Ois200_Data\240718_Speckle_Test'
# preprocess first
One_Key_OIS_Preprocessor(raw_folder)

#%% Get single F frame.
data_folder = r'D:\ZR\_Data_Temp\Ois200_Data\240718_Speckle_Test\Test2_5ms\Preprocessed'
frame = np.load(cf.join(data_folder,'Speckle.npy'))

def get_local_stats(image,width = 1024,length = 1024, window_size=7):
    if image.shape != (width, length):
        raise ValueError("Input image must be 512x512 pixels.")
    
    half_window = window_size // 2
    local_stats = np.zeros((width, length, 2))
    
    for i in tqdm(range(half_window, width - half_window)):
        for j in range(half_window, length - half_window):
            window = image[i - half_window:i + half_window + 1, j - half_window:j + half_window + 1]
            local_stats[i, j, 0] = np.mean(window)
            local_stats[i, j, 1] = np.std(window)
    
    return local_stats

# example_frame = frame.mean(0)
example_frame = frame[403,:,:]
# sns.heatmap(example_frame,center = 0)
local_stats = get_local_stats(example_frame,1024,1024,3)
#%% 
test_graph = local_stats[:,:,1]/local_stats[:,:,0]
sns.heatmap(test_graph,vmin = 0,vmax = 0.05,cmap = 'gist_gray')
#%% Try some GPT codes.
def calculate_speckle_contrast(image_stack, window_size=4):
    """
    Calculates the speckle contrast for a stack of speckle images.
    
    Args:
        image_stack (numpy.ndarray): A 3D numpy array of shape (num_images, 512, 512) containing the speckle image stack.
        window_size (int): The size of the window to use for calculating local statistics (default is 4).
        
    Returns:
        numpy.ndarray: A 2D numpy array of shape (512, 512) containing the speckle contrast map.
    """
    if image_stack.shape[1:] != (1024, 1024):
        raise ValueError("Input image stack must have a shape of (num_images, 512, 512).")
    
    half_window = window_size // 2
    speckle_contrast = np.zeros((1024, 1024))
    
    for i in tqdm(range(half_window, 1024 - half_window)):
        for j in range(half_window, 1024 - half_window):
            window = image_stack[:, i - half_window:i + half_window + 1, j - half_window:j + half_window + 1]
            window_mean = np.mean(window)
            window_std = np.std(window)
            speckle_contrast[i, j] = window_std / window_mean
    
    return speckle_contrast

# Example usage
speckle_contrast_map = calculate_speckle_contrast(frame[1020:1040,:,:], window_size=5)
#%%
fig,ax = plt.subplots(ncols=1,nrows=1,figsize = (5,5),dpi = 180)
sns.heatmap(speckle_contrast_map,vmin = 0,vmax = 0.05,cmap = 'gist_gray',xticklabels=False,yticklabels=False,cbar=False,square=True,ax = ax)
# fig.savefig(cf.join(r'D:\ZR\_Data_Temp\Ois200_Data\240718_Speckle_Test\Test2_5ms\Preprocessed\Speckel_Figs','Test.png'))
#%% We will do scatter for all data points, and save it into a matrix.
winsize = 5
winnum = frame.shape[0]//winsize

graph_path = r'D:\ZR\_Data_Temp\Ois200_Data\240718_Speckle_Test\Test2_5ms\Preprocessed\Speckel_Figs'
contrast_maps = np.zeros(shape = (winnum,1024,1024),dtype = 'f8')
for i in range(winnum):
    c_graph_set = frame[i*winsize:(i+1)*winsize,:,:]
    c_speckle_contrast_map = calculate_speckle_contrast(c_graph_set, window_size=4)
    contrast_maps[i,:,:] = c_speckle_contrast_map
    # plot and save
    plt.clf()
    plt.cla()
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize = (5,5),dpi = 180)
    sns.heatmap(c_speckle_contrast_map,vmin = 0,vmax = 0.04,cmap = 'gist_gray',xticklabels=False,yticklabels=False,cbar=False,square=True,ax = ax)
    fig.savefig(cf.join(graph_path,f'{10000+i}.png'))
    plt.close(fig)
cf.Save_Variable(graph_path,'All_Speckle_Graph',contrast_maps)
#%% Plot thresd parts
for i in tqdm(range(winnum)):
    c_speckle_contrast_map = contrast_maps[i,:,:]
    plt.clf()
    plt.cla()
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize = (5,5),dpi = 180)
    sns.heatmap(c_speckle_contrast_map,vmin = 0,vmax = 0.03,cmap = 'gist_gray',xticklabels=False,yticklabels=False,cbar=False,square=True,ax = ax)
    fig.savefig(cf.join(graph_path,f'{10000+i}.png'))
    plt.close(fig)
#%% Video Generator
plotable = np.clip(contrast_maps,0,0.05)
frames = plotable*255/plotable.max()

save_path = r'D:\ZR\_Data_Temp\Ois200_Data\240718_Speckle_Test\Test2_5ms\Preprocessed'
import cv2
import skvideo.io
import time
width = frames.shape[2]
height = frames.shape[1]
fps = 15
outputfile = cf.join(save_path,'Speckle_Video.avi')   #our output filename
writer = skvideo.io.FFmpegWriter(outputfile, outputdict={
    '-vcodec': 'rawvideo',  #use the h.264 codec
    #  '-vcodec': 'libx264',
    '-crf': '0',           #set the constant rate factor to 0, which is lossless
    '-preset':'veryslow',   #the slower the better compression, in princple, 
    # '-r':str(fps), # this only control the output frame rate.
    '-pix_fmt': 'yuv420p',
    '-vf': "setpts=PTS*{},fps={}".format(25/fps,fps) ,
    '-s':'{}x{}'.format(width,height)
}) 

for frame in tqdm(frames):
    # cv2.imshow('display',frame)
    writer.writeFrame(frame)  #write the frame as RGB not BGR
    # time.sleep(1/fps)

writer.close() #close the writer
cv2.destroyAllWindows()