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
from scipy.stats import pearsonr
import copy
from scipy.spatial.distance import pdist, squareform


class Atlas_Data_Tools(object):
    name = 'Atlas Data Easy Tool'

    def __init__(self,series,bin=4,min_pix = 20):
        self.series = series
        self.std = series.std(0)
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

    def Get_All_Area_Response(self,keep_unilateral = False):
        # This function will average response by area.
        self.Area_Response = pd.DataFrame(columns = ['Area','Function','LR','pixnum','Series'])
        print('Calculating All Brain Area Response...')
        for i,cloc in tqdm(enumerate(self.all_area_name)):
            for j,hemi in enumerate(['L','R']):
                c_mask = self.MG.Get_Mask(cloc,hemi)
                combined_mask = ((self.std*c_mask)!=0)
                if combined_mask.sum()>self.min_pix:
                    c_response = self.series[:,combined_mask].mean(1)
                    c_func = self.Area_Function[cloc]
                    self.Area_Response.loc[len(self.Area_Response),:] = [cloc,c_func,hemi,combined_mask.sum(),c_response]

        # delete unilateral area matrix if required.
        on_area_name = list(set(self.Area_Response['Area']))
        if keep_unilateral == False: 
            for i,c_name in enumerate(on_area_name):
                if (self.Area_Response['Area']==c_name).sum() <2:
                    self.Area_Response = self.Area_Response[self.Area_Response['Area']!= c_name] # drop uni lateral area.

        # after generation, re arrange matrix by LR and function.
        # self.Area_Response = self.Area_Response.sort_values(by=['LR','Function'],ascending=False).reset_index(drop=True)
        part_L = self.Area_Response.groupby('LR').get_group('L').sort_values(by=['Function','Area'],ascending = False)
        part_R = self.Area_Response.groupby('LR').get_group('R').sort_values(by=['Function','Area'],ascending = True)
        self.Area_Response = pd.concat((part_L,part_R),axis = 0).reset_index(drop = True)




    def Combine_Response_Heatmap(self):
        # this function will return heatmap of area response matrix.
        try :
            self.Area_Response
        except NameError:
            print('Area Response not generated yet.')
            self.Get_All_Area_Response()

        # write area response into a pd frame.
        response_heatmap = pd.DataFrame(columns = np.arange(len(self.Area_Response.iloc[0,-1])))
        print('Combining Heatmap of Brain Area Response...')
        for i in tqdm(range(len(self.Area_Response))):
            c_slice = self.Area_Response.iloc[i,:]
            c_name = c_slice['Area']+'_'+c_slice['LR']
            response_heatmap.loc[c_name,:] = c_slice['Series']
        self.Area_Response_Heatmap = response_heatmap.astype('f8')
        return self.Area_Response_Heatmap

    def Get_Corr_Matrix(self,win_size = 600,win_step = 150):

        # this function will calculate corr matrix for area pair we get.
        used_area_response = copy.deepcopy(self.Area_Response)
        used_area_response['Fullname'] = used_area_response['Area']+'_'+used_area_response['LR']
        used_area_fullname = list(used_area_response['Fullname'])

        # get corr matrix for given data frame.
        frame_num = len(self.Area_Response.iloc[0,-1])
        win_num = 1+(frame_num-win_size)//win_step
        self.Corr_Matrix = {}
        print('Calculating Correlation Matrix of given data frames.')
        for i in tqdm(range(win_num)):
            c_start = i*win_step
            c_end = i*win_step+win_size
            c_corr_frame = pd.DataFrame(0.0,columns = used_area_fullname,index = (used_area_fullname))
            for j,name_A in enumerate(used_area_fullname):
                series_A = used_area_response[used_area_response['Fullname']==name_A]['Series'].iloc[0][c_start:c_end]
                for k,name_B in enumerate(used_area_fullname):
                    series_B = used_area_response[used_area_response['Fullname']==name_B]['Series'].iloc[0][c_start:c_end]
                    c_r,_ = pearsonr(series_A,series_B)
                    c_corr_frame.loc[name_A,name_B]=c_r
            self.Corr_Matrix[i] = c_corr_frame
        return self.Corr_Matrix

#%% 
#################################### CLASS END HERE #######################################
# below are functions for contralateral calculation.


def Contra_Similar(series,bin=4):
    '''
    Calculate contralateral similarity of given series. MAKE SURE MASK ALREADY BE DONE.
    '''
    _,height,width = series.shape
    mirrored_series = series[:, :, ::-1]

    MG = Mask_Generator(bin=bin)
    area_mask = MG.idmap>0
    series_mask = series.std(0)>0
    joint_mask = area_mask*series_mask 
    joint_mask = joint_mask*joint_mask[:,::-1] # here are mask where all values are not 0.
    similar_map = np.zeros(shape = (height,width),dtype='f8')

    for i in range(height):
        for j in range(width):
            if joint_mask[i,j] == True:
                c_r,_ = pearsonr(series[:,i,j],mirrored_series[:,i,j])
                similar_map[i,j] = c_r

    return similar_map

######################################Pairwise Corr Function Here.#############################################
def Paiwise_Calculator(matrix, mask):
    '''
    Calculate pairwise correlation relationship inside the mask.
    Input :
        matrix : (N_Frame*Height*width),nd array of graph, usually Z series generated previously.
        mask : (Height*Width) current area mask, must be 1/0 series and the same size as matrix.
    '''
    # Step 1: Index pixels in the mask
    coords = np.argwhere(mask)
    n = coords.shape[0]
    if n == 0:
        raise ValueError("Mask has no active pixels.")
    
    # Step 2: Extract time series data for each pixel
    data_2d = matrix[:,coords[:, 0], coords[:, 1]]
    
    # Compute Pearson correlation matrix
    corr_matrix = np.corrcoef(data_2d.T)
    
    # Step 3: Compute pairwise Euclidean distances
    distances = pdist(coords, metric='euclidean')
    dist_matrix = squareform(distances)
    
    # Step 4: Create DataFrame with upper triangle values
    triu_i, triu_j = np.triu_indices(n, k=1)
    df = pd.DataFrame({
        'ID_A': triu_i + 1,  # Convert to 1-based ID
        'ID_B': triu_j + 1,
        'Distance': dist_matrix[triu_i, triu_j],
        'Correlation': corr_matrix[triu_i, triu_j]
    })
    
    return df

def Pairwise_ID_Loc(mask, pixel_id):
    '''
    Return x,y coordinate of id for pixels in pairwise calculator generated data frame.
    '''
    coords = np.argwhere(mask)
    if pixel_id < 1 or pixel_id > coords.shape[0]:
        raise ValueError("Invalid pixel ID.")
    return coords[pixel_id - 1].astype(int)

#%%
if __name__ == '__main__':
    ADT = Atlas_Data_Tools(series=series,bin=4,min_pix = 100)
    ADT.Area_Function
    ADT.Get_All_Area_Response()