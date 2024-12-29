'''
this script is used for processing speckle data




#########################LOGS################################

including acquire for (1)one speckle map with sliding space
(2) all speckle maps with sliding time
(3) generate a video for speckle maps as a blood flow  result

ver 0.0.1 by WKX, function created. 2024/10/25

'''
#%%
import OI_Functions.Common_Functions as cf
from OI_Functions.Stim_Frame_Align import Stim_Camera_Align
from OI_Functions.Stimulus_dRR_Calculator import dRR_Generator
from OI_Functions.Map_Subtractor import Sub_Map_Generator
import numpy as np
import re
import matplotlib.pyplot as plt
from OIS_Preprocessing import One_Key_OIS_Preprocessor
from OI_Functions.OIS_Preprocessing import Single_Folder_Processor
import seaborn as sns
from tqdm import tqdm
import cv2
import skvideo.io
import time


#(1)one speckle map with sliding space
def calculate_speckle_contrast(image_stack, window_size=4):
    """
    Calculates the speckle contrast for a stack of speckle images.
    
    Args:
        image_stack (numpy.ndarray): A 3D numpy array of shape (num_images, 512, 512) containing the speckle image stack.
        window_size (int): The size of the window to use for calculating local statistics (default is 4).
        
    Returns:
        numpy.ndarray: A 2D numpy array of shape (512, 512) containing the speckle contrast map.
    """
    #if image_stack.shape[1:] != (1024, 1024):
        #raise ValueError("Input image stack must have a shape of (num_images, 512, 512).")
    
    half_window = window_size // 2
    width = image_stack.shape[2]
    height = image_stack.shape[1]
    speckle_contrast = np.zeros((width, height))
    
    for i in tqdm(range(half_window, width - half_window)):
        for j in range(half_window, height - half_window):
            window = image_stack[:, i - half_window:i + half_window + 1, j - half_window:j + half_window + 1]
            window_mean = np.mean(window)
            window_std = np.std(window)
            speckle_contrast[i, j] = window_std / window_mean
    
    return speckle_contrast


#%%
#all speckle maps with sliding time
def generate_speckle_stacks (graph_path,frame,timesize = 5,window_size=4):

    winnum = frame.shape[0]//timesize
    width = frame.shape[2]
    height = frame.shape[1]
    contrast_maps = np.zeros(shape = (winnum,width,height),dtype = 'f8')
    for i in range(winnum):
        c_graph_set = frame[i*timesize:(i+1)*timesize,:,:]
        c_speckle_contrast_map = calculate_speckle_contrast(c_graph_set, window_size)
        contrast_maps[i,:,:] = c_speckle_contrast_map
        # plot and save
        plt.clf()
        plt.cla()
        fig,ax = plt.subplots(ncols=1,nrows=1,figsize = (5,5),dpi = 180)
        sns.heatmap(c_speckle_contrast_map,vmin = 0,vmax = 0.04,cmap = 'gist_gray',xticklabels=False,yticklabels=False,cbar=False,square=True,ax = ax)
        fig.savefig(cf.join(graph_path,f'{10000+i}.png'))
        plt.close(fig)
        cf.Save_Variable(graph_path,'All_Speckle_Graph',contrast_maps)
    return contrast_maps



#%% Video Generator

def generate_speckle_video (path,contrast_maps,fps = 15):
    #plotable = np.clip(contrast_maps,0,255)
    #frames = plotable*255/plotable.max()
    
    #arr_normalized = np.clip(contrast_maps / contrast_maps.max(), 0, 1)
    # 转换为 uint8
    #frames = (arr_normalized * 255).astype(np.uint8)
    frames = (contrast_maps * 255).astype(np.uint8)
    
    #frames = contrast_maps
    width = frames.shape[2]
    height = frames.shape[1]
    outputfile = cf.join(path,'Video.avi')   #our output filename
    writer = skvideo.io.FFmpegWriter(outputfile, outputdict={
    '-vcodec': 'rawvideo',  #use the h.264 codec
    #'-vcodec': 'libx264',  # 使用libx264编码
    #  '-vcodec': 'libx264',
    '-crf': '0',           #set the constant rate factor to 0, which is lossless
    '-preset':'veryslow',   #the slower the better compression, in princple, 
    # '-r':str(fps), # this only control the output frame rate.
    '-pix_fmt': 'yuv420p',
    '-vf': "setpts=PTS*{},fps={}".format(25/fps,fps),
    '-s':'{}x{}'.format(width,height)
    }) 
    for frame in tqdm(frames):
         cv2.imshow('display',frame)
         writer.writeFrame(frame)  #write the frame as RGB not BGR
         # time.sleep(1/fps)
    writer.close() #close the writer
    cv2.destroyAllWindows()
    return outputfile


#%%
if __name__ == "__main__":
    # 测试代码
    graph_path = r'D:\ZR\_Data_Temp\Ois200_Data\240718_Speckle_Test\Test2_5ms\Preprocessed'
    speckle_img = np.load(cf.join(graph_path,'Speckle.npy'))
    contrast_maps = generate_speckle_stacks(graph_path,speckle_img,timesize = 5,window_size=4)
    generate_speckle_video(graph_path,contrast_maps,fps = 15)

#w = cv2.imread(cf.Get_File_Name(graph_path,'.png')[1])
#contrast_maps = np.zeros(shape = (len(cf.Get_File_Name(graph_path,'.png')),w.shape[0],w.shape[1],w.shape[2]),dtype = 'f8')
#for i in range(len(cf.Get_File_Name(graph_path,'.png'))-1):
    #contrast_maps[i,:,:,:] = cv2.imread(cf.Get_File_Name(graph_path,'.png')[i])
