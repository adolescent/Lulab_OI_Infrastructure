'''
This script will make rat drr video, for estimation of whether we have blood-oxygen variation.
'''


#%%

from VDaQ_dRR_Generator import BLK2DRR
import numpy as np
import Common_Functions as cf
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm


wp = r'D:\ZR\_Data_Temp\Rat_Data\RF_0109\cuts'
transfomer = BLK2DRR(wp)
transfomer.Read_All_Frames()
all_graphs = transfomer.all_graphs


#%%
c_frame = all_graphs[-1,2,:,:,:]
c_drr = c_frame/c_frame.mean(0)-1
# c_drr = c_frame

import skvideo.io
def Video_Plotter(series,savepath,filename,fps=25):
    norm_series_A = series/max(series.max(),abs(series.min()))
    plotable_A = (norm_series_A*127+127).astype('u1')
    # concat_series = np.concatenate((plotable_A,plotable_B),axis=2)
    concat_series = plotable_A

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

Video_Plotter(np.clip(c_drr,-0.002,0.002),wp,'drr',4)