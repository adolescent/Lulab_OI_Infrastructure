'''
facemap is a GUI tool, here we check vars they return.
'''



#%%
# import OI_Functions.Common_Functions as cf
import numpy as np
import matplotlib.pyplot as plt

# a = np.load(r'D:\_DataTemp\241213_Awake_Video\cutted_proc.npy',allow_pickle=True)
a = np.load(r'D:\_DataTemp\241213_Awake_Video\full_video_proc.npy',allow_pickle=True)
a = a.flatten()[0]
#%%
pupil_area = a['pupil'][0]['area']
plt.plot(pupil_area[40000:50000])
#%% show 3 whisker's response.
import h5py    
import numpy as np
import pandas as pd

# filename = r'D:\_DataTemp\241213_Awake_Video\cutted_FacemapPose.h5' #origin data value.
filename = r'D:\_DataTemp\241213_Awake_Video\full_video_FacemapPose.h5' #origin data value.

f = h5py.File(filename, 'r+')
dset = f['Facemap']
keys = list(dset.keys())
print(keys)
w1x = np.array(dset['whisker(I)']['x'])
w1y = np.array(dset['whisker(I)']['y'])
w2x = np.array(dset['whisker(II)']['x'])
w2y = np.array(dset['whisker(II)']['y'])
w3x = np.array(dset['whisker(III)']['x'])
w3y = np.array(dset['whisker(III)']['y'])
f.close()
# plot whisker and pupil together.
plt.plot(pupil_area[40000:50000]/10+1000)
# plt.plot(w1x)
plt.plot(w2x[40000:50000])
# plt.plot(w3x)
