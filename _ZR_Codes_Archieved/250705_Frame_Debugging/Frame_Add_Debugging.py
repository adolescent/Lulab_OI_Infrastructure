'''
It seems that frame add in codes will lead to some bugs.. Let's see what happends
'''



#%% import and basic path part
import Common_Functions as cf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm
import pandas as pd
from Brain_Atlas.Atlas_Mask import Mask_Generator
from OI_Functions.OIS_Preprocessing import Single_Folder_Processor # this function is standard bin transformer.
from OIS_Tools import *

datafolder = r'D:\Debugging'

buggy_file = r'D:\Debugging\img_00017.bin'
ok_file = r'D:\Debugging\img_00029.bin'

#%%

with open(buggy_file, mode='rb') as file: # b is important -> binary
    header_bytes = file.read(20) # in Ois system, the first 5 4bit are header info
    data_bytes = file.read()
# read header
header = struct.unpack('5i', header_bytes) # headers are int format(i)
header_info = np.array(header)

total_framenum = len(data_bytes)//header_info[3]
x_width = header_info[1]
y_width = header_info[2]

all_frames = np.zeros(shape = (total_framenum,x_width,y_width),dtype='u2')
all_heads = np.zeros(shape = (total_framenum,3),dtype='i4') # head info of img


for i in range(total_framenum):
    c_graph_bytes = data_bytes[i*header_info[3]:(i+1)*header_info[3]]
    # get each graph head, having it's [index,missed trigger,camera clc.]
    c_graph_head = np.array(struct.unpack(f'3q', c_graph_bytes[:24])) # data are recorded in int64, so 8 bit in 1 unit.
    img_data = np.array(struct.unpack(f'{x_width*y_width}H',c_graph_bytes[24:]))
    all_frames[i,:,:] = img_data.reshape((x_width,y_width))
    all_heads[i,:] = c_graph_head
    all_heads[:,1] = all_heads[:,1]>0
file.close()
del data_bytes

plt.plot(all_heads[:,1])

# # below is adjustment of data frame, all nan to missed trigger location.
# num_adj = int(len(all_frames)+all_heads[:,1].sum())
# all_frames_adj = np.zeros(shape = (num_adj,x_width,y_width),dtype='u2')
# miss_num = all_heads[:,1] # number of missed trigger
# new_index = 0

