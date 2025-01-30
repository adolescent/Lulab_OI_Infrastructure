
'''
Try to do hierarchical cluster on pixels, greedy method of combining nearest 

'''




#%% import 
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
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist,squareform
import time

wp = r'D:\_DataTemp\OIS\Wild_Type\Preprocessed'
MG = Mask_Generator(bin=4)

on_response,on_mask = cf.Load_Variable(wp,'Response_1d.pkl')

#%% 
# Standardize each time series (important for distance metrics)
# on_response = on_response.T
normalized_ts = (on_response - on_response.mean(axis=1, keepdims=True))/on_response.std(axis=1, keepdims=True)

# B. calculate distance matrix.
distance_metric = 'cityblock'  # Multiple options: 'euclidean', 'correlation', 'cityblock','chebyshev'
print("Calculating distance matrix...")
start_time = time.time()
# You need this method to calculate densed dist matrix. it must be dense.
distance_matrix = pdist(normalized_ts , metric=distance_metric)
print(f"Distance matrix computed in {time.time()-start_time:.2f} seconds")
corr_matrix = squareform(distance_matrix) # vice versa,
plt.imshow(corr_matrix)

#%%
# Dendrogram Visualization
Z = linkage(distance_matrix, method='ward',metric=distance_metric )  # multiple method is possible, they will return different results.



# Plot truncated dendrogram for better visibility
fig,ax = plt.subplots(ncols=1,nrows=1,figsize = (6,5),dpi = 180)
a = dendrogram(
    Z,
    truncate_mode='lastp',  # Show only the last p merged clusters
    p=20000,                   # Number of clusters to show
    show_leaf_counts=True,
    leaf_rotation=90,
    leaf_font_size=12,
    show_contracted=True,ax = ax    # Show contracted branches
)
ax.set_title('Hierarchical Clustering Dendrogram (Truncated)')
ax.set_xlabel('Sample Index')
ax.set_ylabel('Distance')
ax.set_xticks([])

# plt.show()
# Z in shape nearest combinationA, combinationB,distance,current element num.
# will do the time of element number.
#%% select specific cluster num or specific 

# Determine cutoff (adjust based on dendrogram inspection)
cutoff_distance =25000  # This is example value - adjust based on your dendrogram!

# Alternatively: Specify number of desired clusters
n_clusters = 8

# Method 1: Distance-based cutoff
# clusters = fcluster(Z, t=cutoff_distance, criterion='distance')
# Method 2: Number of clusters
clusters = fcluster(Z, t=n_clusters, criterion='maxclust')

# recover cluster to mask
recover = copy.deepcopy(on_mask).astype('i4')
recover[recover==True] = clusters
# sns.heatmap(recover,cmap='rainbow',square=True)
plt.imshow(recover,cmap='jet')




#%% temporal visualize
c_vec = on_response[clusters==12].mean(0)
recover = copy.deepcopy(on_mask).astype('f8')
recover[on_mask==True] = c_vec
sns.heatmap(recover,square=True,center=0)

# plt.imshow(recover,cmap='jet')