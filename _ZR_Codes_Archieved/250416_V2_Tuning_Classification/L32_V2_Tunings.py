'''
This script will try to calculate tunings of orientation, direction, and color tuning preference.

Then we will try to calculate the pixel's distance to V1-V2 boulder & to Lunate , making it possible for vertical calculation.

The horizontal distance is also important.

And at last use this to classify all V2 pixels into 3 class.

'''

#%%
import Common_Functions as cf
from OI_Functions.Map_Subtractor import Sub_Map_Generator
from OI_Functions.VDaQ_dRR_Generator import BLK2DRR
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2
import pandas as pd

wp = r'D:\ZR\_Data_Temp\VDaQ_Data\200910_L80_LM_OI\Run01_G8'

# show blk file names on example folder.
cf.Get_File_Name(wp,'.BLK')

#%%



