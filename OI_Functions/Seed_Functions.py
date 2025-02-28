'''
These functions will work on seed point correlation.

'''


#%%
import Common_Functions as cf
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from Signal_Functions.Filters import Signal_Filter_1D
from scipy import stats
import seaborn as sns
from OI_Functions.VDaQ_dRR_Generator import BLK2DRR
from scipy.stats import pearsonr

def Generate_Circle_Mask(center,radius,height,width):
    y, x = np.ogrid[:height, :width]
    dist = (x - center[1])**2 + (y - center[0])**2
    circle_mask = dist <= radius**2
    return circle_mask


def Seed_Corr_Core(seed_series,response_matrix):
    
    heights = response_matrix.shape[1]
    widths = response_matrix.shape[2]
    corr_matrix = np.zeros(shape = (heights,widths))
    for i in range(heights):
        for j in range(widths):
            c_pix = response_matrix[:,i,j]
            if c_pix.any(): # not all zero series
                c_corr,_ = pearsonr(c_pix,seed_series)
            else:
                c_corr = np.nan
            corr_matrix[i,j] = c_corr

    return corr_matrix


def Seed_Window_Slide(seed_mask,response_matrix,win_size,win_step):
    seed_series = response_matrix[:,seed_mask].mean(1)
    if len(response_matrix) < (win_size+win_step):
        winnum = 1
    else:
        winnum = (len(response_matrix)-win_size)//win_step
    corr_wins = np.zeros(shape = (winnum,response_matrix.shape[1],response_matrix.shape[2]),dtype='f8')
    for i in tqdm(range(winnum)):
        c_matrix = response_matrix[i*win_step:win_size+i*win_step,:,:]
        c_seed = seed_series[i*win_step:win_size+i*win_step]
        c_corr_matrix = Seed_Corr_Core(c_seed,c_matrix)
        corr_wins[i,:,:] = c_corr_matrix
    return corr_wins




#%% Test Run Parts

if __name__ == '__main__':
    wp = r'D:\ZR\_Data_Temp\VDaQ_Data\241015_Niid_Spon\Run03_SPON'
    r_detrend = cf.Load_Variable(wp,'R_detrend.pkl')
    drr_train = r_detrend/r_detrend.mean(0)-1
    # drr_train_raw = r_train/r_train.mean(0)-1
    response_matrix = drr_train[:]
    height = r_detrend.shape[1]
    width = r_detrend.shape[2]
    
    seed_coords = (210,280) # height,width
    r = 10 # radius of seed circle.
    seed_mask = Generate_Circle_Mask(seed_coords,r,height,width)
    corr_wins = Seed_Window_Slide(seed_mask,response_matrix,1200,1200)

    # save slide window corrs
    savepath = cf.join(wp,'Corrs')
    cf.mkdir(savepath)
    for i in range(len(corr_wins)):
        c_corr = corr_wins[i,:,:]
        plt.clf()
        plt.cla()
        fig,ax = plt.subplots(ncols=1,nrows=1,figsize = (5,5),dpi = 240)
        sns.heatmap(c_corr,xticklabels=False,yticklabels=False,square=True,ax = ax,center=0,vmax = 1,vmin =-1)
        fig.savefig(cf.join(savepath,f'{10000+i}.png'))
        plt.close(fig)

