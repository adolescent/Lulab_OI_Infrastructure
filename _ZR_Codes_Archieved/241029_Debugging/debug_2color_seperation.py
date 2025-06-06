'''
We find that we have problems on color seperation, but ois works fine. Debugging here.

'''

#%%
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import OI_Functions.Common_Functions as cf
from OI_Functions.OIS_Tools import *
import numpy as np
from scipy.io import savemat
import time


path = r'E:\20241010_KCL_CSD\20241010_CSD_1 M_2mm\red_speckle'

#%%
info_path = cf.Get_File_Name(path,'.txt','info')[0]
analog_file_names = cf.Get_File_Name(path,'.bin','ai')
img_file_names = cf.Get_File_Name(path,'.bin','img')

channel_names = Info_Reader(info_path)['Channel_Names']
_,graphs_all = Graph_Reader_All(img_file_names,channel_names)

#%% It seems that the bug origin before concat, so single blk will solve the problem.
filename = r'E:\20241010_KCL_CSD\20241010_CSD_1 M_2mm\red_speckle\img_00000.bin'
with open(filename, mode='rb') as file: # b is important -> binary
    header_bytes = file.read(20) # in Ois system, the first 5 4bit are header info
    data_bytes = file.read()

header = struct.unpack('5i', header_bytes) # headers are int format(i)
header_info = np.array(header)

# unpack data. For each frame, it have 3 int64 headers.
total_framenum = len(data_bytes)//header_info[3]
x_width = header_info[1]
y_width = header_info[2]

all_frames = np.zeros(shape = (total_framenum,x_width,y_width),dtype='u2')
all_heads = np.zeros(shape = (total_framenum,3),dtype='i4')

for i in tqdm(range(total_framenum)):
    c_graph_bytes = data_bytes[i*header_info[3]:(i+1)*header_info[3]]
    # get each graph head, having it's [index,missed trigger,camera clc.]
    c_graph_head = np.array(struct.unpack(f'3q', c_graph_bytes[:24])) # data are recorded in int64, so 8 bit in 1 unit.
    img_data = np.array(struct.unpack(f'{x_width*y_width}H',c_graph_bytes[24:]))
    all_frames[i,:,:] = img_data.reshape((x_width,y_width))
    all_heads[i,:] = c_graph_head

file.close()
del data_bytes

plt.plot(all_heads[:,1])
#%% we find problems here! the missed trigged caused alteration of graphs.
# so we only need to insert nan frame into the data matrix, and all will be done.
num_adj = int(len(all_frames)+all_heads[:,1].sum())
all_frames_ajusted = np.zeros(shape = (num_adj,x_width,y_width),dtype='u2')
miss_num = all_heads[:,1]


new_index = 0
for i in tqdm(range(len(miss_num))):

    # Insert NaN images based on miss_num
    for _ in range(miss_num[i]):
        all_frames_ajusted[new_index] = np.full((256, 256), np.nan)
        new_index += 1

    # Insert the original frame
    all_frames_ajusted [new_index] = all_frames[i]
    new_index += 1
    # new_index += 1

# del all_frames


#%% get all odd and end frames
odd_frames = frame[1::2] 
# end_frames = all_frames[0::2] 
plt.plot(odd_frames.mean(-1).mean(-1))

#%%
from OIS_Preprocessing import Single_Folder_Processor

Single_Folder_Processor(r'E:\20241010_KCL_CSD\20241010_CSD_1 M_2mm\red_speckle','python','Test')


