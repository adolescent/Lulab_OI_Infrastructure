'''
This script provide several tools for atlas data processing.
Before input, chamber mask shall be done. otherwise, result might not be accurate.

'''
#%%
from Brain_Atlas.Atlas_Mask import Mask_Generator
import numpy as np
import matplotlib.pyplot as plt
import Common_Functions as cf
import pandas as pd
from tqdm import tqdm



class Atlas_Data_Tools(object):
    name = 'Atlas Data Easy Tool'

    def __init__(self,series,bin=4,min_pix = 20):
        self.series = series
        self.avr = series.mean(0)
        self.MG = Mask_Generator(bin=4)
        self.all_area_name = self.MG.all_areas
        self.Area_Function = {}
        self.min_pix = min_pix
        for i,c_area in enumerate(self.all_area_name):
            if 'VI' in c_area:
                self.Area_Function[c_area] = 'Visual'
            elif 'SS' in c_area:
                self.Area_Function[c_area] = 'Somatosensory'
            elif 'MO' in c_area:
                self.Area_Function[c_area] = 'Motion'
            elif 'AUD' in c_area:
                self.Area_Function[c_area] = 'Auditory'
            elif 'RSP' in c_area:
                self.Area_Function[c_area] = 'Retro'
            else:
                self.Area_Function[c_area] = 'Other'

    def Get_All_Area_Response(self):
        # This function will average response by area.
        self.Area_Response = pd.DataFrame(columns = ['Area','Function','LR','pixnum','Series'])
        for i,cloc in tqdm(enumerate(self.all_area_name)):
            for j,hemi in enumerate(['L','R']):
                c_mask = self.MG.Get_Mask(cloc,hemi)
                combined_mask = ((self.avr*c_mask)!=0)
                if combined_mask.sum()>self.min_pix:
                    c_response = self.series[:,combined_mask].mean(1)
                    c_func = self.Area_Function[cloc]
                    self.Area_Response.loc[len(self.Area_Response),:] = [cloc,c_func,hemi,combined_mask.sum(),c_response]

        # after generation, re arrange matrix by LR and function.
        # self.Area_Response = self.Area_Response.sort_values(by=['LR','Function'],ascending=False).reset_index(drop=True)
        part_L = self.Area_Response.groupby('LR').get_group('L').sort_values(by=['Function'],ascending = False)
        part_R = self.Area_Response.groupby('LR').get_group('R').sort_values(by=['Function'],ascending = True)
        self.Area_Response = pd.concat((part_L,part_R),axis = 0).reset_index(drop = True)




    def Combine_Response_Heatmap(self):
        # this function will return heatmap of area response matrix.
        try :
            self.Area_Response
        except NameError:
            print('Area Response not generated yet.')
            self.Get_All_Area_Response()








#%%
if __name__ == '__main__':
    ADT = Atlas_Data_Tools(series=series,bin=4,min_pix = 100)
    ADT.Area_Function
    ADT.Get_All_Area_Response()