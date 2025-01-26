'''
Redo id map, fill overlapping of idmap and mask map.
'''


#%% Import
import Common_Functions as cf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Atlas_Mask import Mask_Generator

MG = Mask_Generator(bin=2)
masks = MG.masks
idmap = MG.idmap

#%% get over lapping idmap.
full_mask = np.zeros(shape = idmap.shape,dtype='i4')

for i in range(len(masks)):
    c_mask = masks.iloc[i,-1]
    full_mask+=c_mask

inner_boulders = full_mask>1
outer_boulders = full_mask>0

# left inner boulder to blank.
# idmap[inner_boulders]=0
plt.imshow(full_mask)
#%% fix idmap, fill only unfilled location.
idmap_new = np.zeros(shape = idmap.shape,dtype='i4')
writable_mask = np.ones(shape = idmap.shape,dtype='i4')

for i in range(len(masks)):
    c_mask = masks.iloc[i,-1]
    c_id = masks.iloc[i,0]
    c_mask_joint = writable_mask*c_mask
    writable_mask-= c_mask
    writable_mask = writable_mask.clip(0,1)
    idmap_new+= c_mask_joint*c_id
np.save('Raw_Area_ID_bin2.npy',idmap_new)
#%% after this, we fix mask var with new id.
import copy
import os

masks_new = copy.deepcopy(masks)

for i in range(len(masks)):
    # c_mask = masks_new.iloc[i,-1]
    c_id = masks_new.iloc[i,0]
    c_mask_redo = idmap_new==c_id
    masks_new['Mask'].iloc[i] = c_mask_redo
cf.Save_Variable(os.getcwd(),'Brain_Area_Masks_bin2',masks_new)


#%% check full mask after process, here will be no overlapping.
full_mask = np.zeros(shape = idmap.shape,dtype='i4')

for i in range(len(masks_new)):
    c_mask = masks_new.iloc[i,-1]
    full_mask+=c_mask

inner_boulders = full_mask>1
outer_boulders = full_mask>0

# plt.imshow(inner_boulders)
plt.imshow(full_mask)