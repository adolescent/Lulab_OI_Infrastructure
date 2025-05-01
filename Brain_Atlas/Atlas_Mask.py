

#%%
import seaborn as sns
import Common_Functions as cf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
import copy
import cv2

class Mask_Generator(object):

    name = 'Mask of visable parts'

    def __init__(self,bin=4) -> None:# bin is the only require od 
        base_path = os.path.dirname(__file__)
        breg = cf.Load_Variable(cf.join(base_path,'Bregma.pkl')) # breg in seq y,x
        # self.breg = 
        # load area map and mask for current bin. method below is dumb, but it works.
        # NOTE self.idmap may not return perfect map to you, you will need self.masks
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

    def ID_Name(self,c_id):
        # inner function, input an ID of idmap, return its' name.
        c_area_name = self.masks.loc[self.masks['ID'] == c_id]['Area'].iloc[0]
        c_hemi = self.masks.loc[self.masks['ID'] == c_id]['LR'].iloc[0]
        whole_name = c_area_name+'_'+c_hemi
        return whole_name

    def Avr_By_Area(self,graph,min_pix = 100):
        # add up a input graph, and return brain-area response matrix.
        # NOTE you can use this function to average ANY pix map.

        # area with pix less than min_pix will be ignored.
        graph_mask = (graph != 0)
        # get all id of given mask

        all_id = np.arange(1,63) # all visable id on 
        avr_response = pd.DataFrame(columns = ['Name','Response'])
        for i,c_id in enumerate(all_id):
            c_mask = self.masks.loc[self.masks['ID']==c_id]['Mask'].iloc[0]
            c_name = self.ID_Name(c_id)
            joint_mask = graph_mask*c_mask
            # avr graph if value is ok.
            if joint_mask.sum()>min_pix:
                c_response = graph[joint_mask==True].mean()
                avr_response.loc[len(avr_response),:] = [c_name,c_response]

        return avr_response

    def Get_Weight_Map(self,area_names,weight_frame,spliter = '_'):
        # NOTE elements in area_names shall be in format area_Hemi. e.g. 'VISrl_L'
        # weight frame is the weight of each brain area frame.
        area_names = list(area_names)
        weight_frame = list(weight_frame)
        if len(area_names) != len(weight_frame):
            raise ValueError('Name and weight have different length!')

        weight_map = np.zeros(shape = self.idmap.shape)
        # all_names = list(weight_frame['Name'])
        for i,c_name_full in enumerate(area_names):
            c_weight = weight_frame[i]
            c_area,c_hemi = c_name_full.split(spliter)
            weight_map += c_weight*self.Get_Mask(c_area,c_hemi).astype('f8')
            
        return weight_map

    def Get_Func_Mask(self,area = 'VI',LR='L'):
        # area can be VI/SS/MO/RSP, indicating visual, somatosensory, motion and retrosplenial area.
        # this function will combine functional brain areas.
        c_mat = self.masks.groupby('LR').get_group(LR)
        c_mat_names = list(c_mat['Area'])
        combined_mask = np.zeros(shape=self.idmap.shape,dtype='bool')
        for i,c_name in enumerate(c_mat_names):
            if area in c_name:
                combined_mask += c_mat[c_mat['Area']==c_name].iloc[0]['Mask']
        return combined_mask
    
    def Area_Counters(self):
        boulders = cv2.Canny(self.idmap.astype('u1'), 0, 1)
        return boulders
    
#%% test tun
if __name__ == '__main__':
    MG = Mask_Generator(bin = 4)
    # plt.imshow(MG.idmap)
    # all_areas = MG.all_areas
    # # mask = MG.Get_Mask()
    # MG.Pix_Label(220,225)
    # c_mask = MG.Get_Mask('VISp','R')
    # plt.imshow(c_mask)
    # print(MG.ID_Name(1))
    plt.imshow(MG.Get_Func_Mask('RSP','L'))