

#%%
import seaborn as sns
import Common_Functions as cf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
import copy

class Mask_Generator(object):

    name = 'Mask of visable parts'

    def __init__(self,bin=4) -> None:# bin is the only require od 
        base_path = os.path.dirname(__file__)
        breg = cf.Load_Variable(cf.join(base_path,'Bregma.pkl')) # breg in seq y,x
        # self.breg = 
        # load area map and mask for current bin. method below is dumb, but it works.
        if bin == 1:
            self.breg = breg
            self.idmap = np.load(cf.join(base_path,'Raw_Area_ID.npy'))
            self.masks = cf.Load_Variable(cf.join(base_path,'Brain_Area_Masks.pkl'))
        elif bin == 2:
            self.breg = (int(breg[0]/2),int(breg[1]/2))
            self.idmap = np.load(cf.join(base_path,'Raw_Area_ID_bin2.npy'))
            self.masks = cf.Load_Variable(cf.join(base_path,'Brain_Area_Masks_bin2.pkl'))
        elif bin == 4:
            self.breg = (int(breg[0]/4),int(breg[1]/4))
            self.idmap = np.load(cf.join(base_path,'Raw_Area_ID_bin4.npy'))
            self.masks = cf.Load_Variable(cf.join(base_path,'Brain_Area_Masks_bin4.pkl'))
        else:
            raise IOError(f'Bin {bin} not supported.')
        self.all_areas = list(set(self.masks['Area']))

    def Pix_Label(self,y,x): # insequence Y,X
        
        c_id = self.idmap[y,x]
        if c_id == 0:
            print('Pixel out of visible cortex.')
            area_name = 'Outside'
            hemi = 'Both'
        else:
            area_name = self.masks.loc[self.masks['ID'] == c_id]['Area'].iloc[0]
            hemi = self.masks.loc[self.masks['ID'] == c_id]['LR'].iloc[0]
            print(f'Pixel in area {area_name}, on hemisphere {hemi}')

        return area_name,hemi


    def Get_Mask(self,area,LR = 'L'):
        hemi = self.masks.groupby('LR').get_group(LR)
        c_mask = hemi.loc[hemi['Area']==area]['Mask'].iloc[0]
        return c_mask

    

    
#%% test tun
if __name__ == '__main__':
    MG = Mask_Generator(bin = 4)
    # plt.imshow(MG.idmap)
    all_areas = MG.all_areas
    # mask = MG.Get_Mask()
    MG.Pix_Label(220,225)
    c_mask = MG.Get_Mask('VISp','R')
    plt.imshow(c_mask)
