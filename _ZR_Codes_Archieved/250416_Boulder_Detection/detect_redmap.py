'''
Red map have too little features, I want to detect boulders for it.

'''

#%%
import Common_Functions as cf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm
import pandas as pd
from Brain_Atlas.Atlas_Mask import Mask_Generator
from Atlas_Corr_Tools import *
from scipy.stats import pearsonr
from Signal_Functions.Pattern_Tools import Do_PCA
import time

wp = r'D:\_DataTemp\OIS\Wild_Type\Preprocessed'
series = np.load(cf.join(wp,'binned_r_series.npy'))
avr = series.mean(0)
#%%
# sns.heatmap(avr)


def detect_boulders_canny(gray, canny_low=50, canny_high=150):
    # Read the image

    
    # Reduce noise with Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges using Canny
    edges = cv2.Canny(blurred, canny_low, canny_high)
    
    # Dilate edges to connect gaps between fragmented edges
    # kernel = np.ones((3, 3), np.uint8)
    # dilated = cv2.dilate(edges, kernel, iterations=2)
    
    return edges

# Example usage
gray = (avr/256).astype('u1')
# high canny_high will return a high resolution.
mask = detect_boulders_canny(gray,canny_low=10, canny_high=120)

plt.imshow(mask)
