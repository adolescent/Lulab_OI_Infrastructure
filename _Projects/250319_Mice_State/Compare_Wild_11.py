'''
Load and concat frames of GGC11 and wild-type mice.

'''

#%%
import Common_Functions as cf
import numpy as np
import matplotlib.pyplot as plt
from Brain_Atlas.Atlas_Mask import Mask_Generator
import seaborn as sns
import cv2
from tqdm import tqdm
import pandas as pd
from Atlas_Corr_Tools import *
import copy
from scipy.stats import pearsonr,ttest_ind


# load frame
wp_wild = r'D:\ZR\_Data_Temp\Ois200_Data\activity_analysis\47#'
wp_11 = r'D:\ZR\_Data_Temp\Ois200_Data\activity_analysis\A6'

corr_info_same_hemi_11 = cf.Load_Variable(wp_11,'Corr_by_Motion_Same.pkl')
corr_info_same_hemi_wild = cf.Load_Variable(wp_wild,'Corr_by_Motion_Same.pkl')
corr_info_same = pd.concat([corr_info_same_hemi_11,corr_info_same_hemi_wild])
corr_info_same['Area_Pair'] = corr_info_same['Area_A']+'-'+corr_info_same['Area_B']

#%% compare diff.
# example_frame = copy.deepcopy(corr_info_same.groupby(['Area_A','Area_B']).get_group(('SS','MO')))
example_frame = copy.deepcopy(corr_info_same.groupby('Motion_Level').get_group('Low'))
example_frame['Corr'] = example_frame['Corr'].astype('f8')

fig,ax = plt.subplots(ncols=1,nrows=1,figsize = (5,4),dpi=240)
sns.boxplot(data=example_frame,x = 'Area_Pair',y='Corr',hue = 'Case_Type',ax = ax,width=0.5,showfliers=False)
# ax.set_ylim(0.85,1)

#%% stat of 2 distribution.
test_pair = example_frame.groupby('Area_Pair').get_group('SS-MO')

resp_wild = test_pair.groupby('Case_Type').get_group('Wild')['Corr']
resp_11 = test_pair.groupby('Case_Type').get_group('GGC11')['Corr']

t,p = ttest_ind(resp_wild,resp_11)
print(f'T test return:t={t:.3f},p={p:.4f}')