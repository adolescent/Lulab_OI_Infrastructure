

#%% import 

import matplotlib.pyplot as plt
import numpy as np
import struct

buggy_file = r'D:\Debugging\img_00017.bin'

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
all_heads = np.zeros(shape = (total_framenum,3),dtype='i8') ## Bug still here after setting int64.


for i in range(total_framenum):
    c_graph_bytes = data_bytes[i*header_info[3]:(i+1)*header_info[3]]
    # get each graph head, having it's [index,missed trigger,camera clc.]
    c_graph_head = np.array(struct.unpack(f'3q', c_graph_bytes[:24])) # data are recorded in int64, so 8 bit in 1 unit.
    # if c_graph_head[1] != 0:
    #     print(c_graph_head)
    img_data = np.array(struct.unpack(f'{x_width*y_width}H',c_graph_bytes[24:]))
    all_frames[i,:,:] = img_data.reshape((x_width,y_width))
    all_heads[i,:] = c_graph_head
    # all_heads[:,1] = all_heads[:,1]>0 # an emergency fix.
file.close()
del data_bytes

plt.plot(all_heads[:,1])


