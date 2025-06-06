'''
We find that the affine have problem of grid-like signal.
This is called Moiré pattern.
We will first try whether other intrapolate method will solve the problem.
If not, we try to do spatial FFT Filter(2D FFT low pass), trying to remove the signal causing problem.


'''
#%%

import numpy as np
import Common_Functions as cf
import matplotlib.pyplot as plt
from Align_Tools import Match_Pattern
import seaborn as sns
import pandas as pd
from Brain_Atlas.Atlas_Mask import Mask_Generator
import cv2

wp = r'D:\ZR\_Data_Temp\Ois200_Data\Affine_Fix'
#%%

raw_r_matrix = np.load(cf.join(wp,'Red.npy'))
used_r_matrix = raw_r_matrix[:18000,:,:]
del raw_r_matrix
# bin and cut r value matrix. capture in 30hz,so bin every 6.
binned_r_matrix = np.reshape(used_r_matrix,(-1,6,512,512)).mean(1).astype('u2')
avr = binned_r_matrix.mean(0)
np.save(cf.join(wp,'Binned_R_10min'),binned_r_matrix)
#%% rotate graph.
series = np.load(cf.join(wp,'Binned_R_10min.npy'))
avr = series.mean(0)
#%%
MP = Match_Pattern(avr = avr,bin = 4,lbd = 4.2)
MP.Select_Anchor()
MP.Fit_Align_Matrix()
#%%
stacks = MP.Transform_Series(stacks=series,intra = cv2.INTER_NEAREST)
np.save(cf.join(wp,'Aligned_Series_INTER_NEAREST'),stacks)
#%% Corr part, let's see several area to find corr bugs.
from Signal_Functions.Filters import  Signal_Filter_1D
from tqdm import tqdm

num,height,width = stacks.shape

HP_freq = 0.01
LP_freq = 1
fps = 5
r_filted = np.zeros(shape = (stacks.shape),dtype='f8')

for i in tqdm(range(height)):
    for j in range(width):
        x = stacks[:,i,j]
        if x.sum()!= 0:
            filted_x = Signal_Filter_1D(x,HP_freq,LP_freq,fps)
            r_filted[:,i,j] = filted_x

drr = np.nan_to_num(r_filted/r_filted.mean(0)-1)
# plt.plot(drr[:,225,75])
# np.save(cf.join(wp,'drr'),drr)
#%% do corr.
from Seed_Functions import *

MG = Mask_Generator(bin=4)
fps = 5
win_len = 120*fps
win_step = 30*fps

# seed_mask = MG.Get_Mask('VISp','L')
seed_mask = Generate_Circle_Mask((225,75),5,330,285)
corr_wins = Seed_Window_Slide(seed_mask,drr,win_len,win_step)
#%% show graph.
savepath = cf.join(wp,'Corrs')
cf.mkdir(savepath)
for i in range(len(corr_wins)):
    c_corr = corr_wins[i,:,:]
    plt.clf()
    # plt.cla()
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize = (5,5),dpi = 240)
    sns.heatmap(c_corr,xticklabels=False,yticklabels=False,square=True,ax = ax,center=0.7,vmax = 1,vmin =0.5)
    fig.savefig(cf.join(savepath,f'{10000+i}.png'))
    plt.close(fig)