'''
This part will generate standarized data for V2 processing.


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



wp = r'D:\ZR\_Data_Temp\VDaQ_Data\200910_L80_LM_OI'
# get run folder of G8 and RGLunm Run.
orien_folder = r'D:\ZR\_Data_Temp\VDaQ_Data\200910_L80_LM_OI\Run01_G8'
color_folder = r'D:\ZR\_Data_Temp\VDaQ_Data\200910_L80_LM_OI\Run12_RGLum4'
## load mask
chamber_mask = cv2.imread(r'D:\ZR\_Data_Temp\VDaQ_Data\200910_L80_LM_OI\_V2_Sripe_Method\Masks\chambermask.bmp',0)>0
orien_mask = cv2.imread(r'D:\ZR\_Data_Temp\VDaQ_Data\200910_L80_LM_OI\_V2_Sripe_Method\Masks\orien_mask.png',0)>0
color_mask = cv2.imread(r'D:\ZR\_Data_Temp\VDaQ_Data\200910_L80_LM_OI\_V2_Sripe_Method\Masks\colormask.png',0)>0
v2_mask = cv2.imread(r'D:\ZR\_Data_Temp\VDaQ_Data\200910_L80_LM_OI\_V2_Sripe_Method\Masks\V2_Mask.png',0)>0
## load drr
color_drr = cf.Load_Variable(r'D:\ZR\_Data_Temp\VDaQ_Data\200910_L80_LM_OI\Run12_RGLum4\Processed\dRR_Dictionaries.pkl')
orien_drr = cf.Load_Variable(r'D:\ZR\_Data_Temp\VDaQ_Data\200910_L80_LM_OI\Run01_G8\Processed\dRR_Dictionaries.pkl')


#%%
'''
Generate normalized response matrix.
Clip and normalize graph with given std,then filt it with Space filter.


'''
from scipy.ndimage import gaussian_filter
import scipy.stats as stats

clip_std = 3
HP_sigma=20
LP_sigma=0.75
color_id = [1,2,3,4] # 5 is blank
color_resp_filted = {}
orien_id = [1,2,3,4,5,6,7,8] # 9 is blank
orien_resp_filted = {}
used_frames = [5,6,7,8,9,10,11,12,13]

# fill color response graph
for i,cid in enumerate(color_id):
    ## blank response
    color_blank = color_drr[5][:,used_frames,:,:].mean(1)
    all_blank = np.zeros(shape = color_blank.shape)
    for j in tqdm(range(len(color_blank))):
        c_blank = color_blank[j,:,:]
        clipped_blank = c_blank.clip(c_blank.mean()-clip_std*c_blank.std(),c_blank.mean()+clip_std*c_blank.std())
        clipped_blank = clipped_blank/clipped_blank.flatten().std()
        # print(clipped_blank.flatten().std())
        HP_graph = gaussian_filter(input = clipped_blank[:,:],sigma=HP_sigma)
        LP_graph = gaussian_filter(input = clipped_blank[:,:],sigma=LP_sigma)
        all_blank[j,:,:] = np.clip(LP_graph-HP_graph,-3,3)

    ## condition resposne
    cc_frame = color_drr[cid]
    avr_response = cc_frame[:,used_frames,:,:].mean(1)
    all_color = np.zeros(shape = avr_response.shape)
    for j in tqdm(range(len(avr_response))):
        c_resp = avr_response[j,:,:]
        clipped_graph = c_resp.clip(c_resp.flatten().mean()-clip_std*c_resp.flatten().std(),c_resp.flatten().mean()+clip_std*c_resp.flatten().std())
        clipped_graph = clipped_graph/clipped_graph.flatten().std()
        HP_graph = gaussian_filter(input = clipped_graph,sigma=HP_sigma)
        LP_graph = gaussian_filter(input = clipped_graph,sigma=LP_sigma)
        filted_graph = np.clip(LP_graph-HP_graph,-3,3)
        all_color[j,:,:] = filted_graph
    ## ttest response
    color_resp_filted[cid],_ = stats.ttest_ind(all_color,all_blank,axis=0)
    # color_resp_filted[cid] = all_color.mean(0)-all_blank.mean(0)


# fill orientation response graph
for i,cid in enumerate(orien_id):
    ## blank response
    color_blank = orien_drr[9][:,used_frames,:,:].mean(1)
    all_blank = np.zeros(shape = color_blank.shape)
    for j in tqdm(range(len(color_blank))):
        c_blank = color_blank[j,:,:]
        clipped_blank = c_blank.clip(c_blank.mean()-clip_std*c_blank.std(),c_blank.mean()+clip_std*c_blank.std())
        clipped_blank = clipped_blank/clipped_blank.flatten().std()
        HP_graph = gaussian_filter(input = clipped_blank[:,:],sigma=HP_sigma)
        LP_graph = gaussian_filter(input = clipped_blank[:,:],sigma=LP_sigma)
        all_blank[j,:,:] = np.clip(LP_graph-HP_graph,-3,3)

    ## condition resposne
    cc_frame = orien_drr[cid]
    avr_response = cc_frame[:,used_frames,:,:].mean(1)
    all_orien = np.zeros(shape = avr_response.shape)
    for j in tqdm(range(len(avr_response))):
        c_resp = avr_response[j,:,:]
        clipped_graph = c_resp.clip(c_resp.flatten().mean()-clip_std*c_resp.flatten().std(),c_resp.flatten().mean()+clip_std*c_resp.flatten().std())
        clipped_graph = clipped_graph/clipped_graph.flatten().std()
        HP_graph = gaussian_filter(input = clipped_graph,sigma=HP_sigma)
        LP_graph = gaussian_filter(input = clipped_graph,sigma=LP_sigma)
        filted_graph = np.clip(LP_graph-HP_graph,-3,3)
        all_orien[j,:,:] = filted_graph
    ## ttest response
    orien_resp_filted[cid],_ = stats.ttest_ind(all_orien,all_blank,axis=0)
    # orien_resp_filted[cid] = all_orien.mean(0)-all_blank.mean(0)

#%% make response matrix here.
# use id 1-8 as orien, id 9-12 as color.
response_matrix = pd.DataFrame(0.0,columns = np.arange(1,13),index = np.arange(v2_mask.sum()))

# fill orien
for i,cid in enumerate(orien_id):
    c_resp = orien_resp_filted[cid]
    response_matrix.loc[:,cid]=c_resp[v2_mask]

for i,cid in enumerate(color_id):
    c_resp = color_resp_filted[cid]
    response_matrix.loc[:,cid+8]=c_resp[v2_mask]

cf.Save_Variable(wp,'Response_Matrix',response_matrix)

#%% recover method
a = np.zeros(shape = v2_mask.shape)
a[v2_mask==True] = point_coords[:,7]
sns.heatmap(a,center=0)

#%%

'''
After Response Matrix, here we will try to calculate PCA for given series.

'''
from Signal_Functions.Pattern_Tools import Do_PCA

pcable_data = (response_matrix/response_matrix.max()).clip(-1,1).T

PC_comps,point_coords,pca = Do_PCA(pcable_data,feature='Time',pcnum=10)
plt.plot(np.cumsum(pca.explained_variance_ratio_))

#%% Plot scatter plot of all points
used_pc = [0,1,2]
# u = point_coords[:10000,used_pc]
u = PC_comps[used_pc,:10000]

fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': '3d'},figsize = (5,5),dpi = 180)
ax.grid(False)
# scatter = ax.scatter(u[:,0],u[:,1],u[:,2],alpha=0.7,edgecolors='none',s=3)
scatter = ax.scatter(u[0,:],u[1,:],u[2,:],alpha=0.7,edgecolors='none',s=3)

#%%
'''
Try some cluster method, see if we can get the correlated frames.

'''
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist,squareform

distance_metric = 'cityblock'
distance_matrix = pdist(PC_comps.T, metric=distance_metric)
distance_matrix.shape
#%% make dendrogram
Z = linkage(distance_matrix, method='ward',metric=distance_metric)

print(Z.shape)
print(Z[0])
fig,ax = plt.subplots(ncols=1,nrows=1,figsize = (4,6),dpi = 180)
a = dendrogram(
    Z,
    truncate_mode='lastp',  # Show only the last p merged clusters
    p=2500,                   #Steps of cluster to show. set very big will show whole dentrogram.
    show_leaf_counts=True,
    leaf_rotation=90, # rotation angle of leaf 
    leaf_font_size=6, # fontsize of leaf
    ax = ax,
    orientation= 'top' # Orientation of cluster going on. top as default.
)
ax.set_title('Dendrogram')
ax.set_xlabel('Samples')
ax.set_ylabel('Distance')
ax.set_xticks([]) # Mute x label if you want full graph.

#%% getting clusters
n_clusters = 30
clusters = fcluster(Z, t=n_clusters, criterion='maxclust')
print(clusters) # cluster is a list of id, indicating the cluster of each pix.

# then visualize.
recover = np.zeros(shape = v2_mask.shape).astype('i4')
recover[v2_mask==True] = clusters
plt.imshow(recover,cmap='jet')

'''
This method seems to be buggy, DECREPTED= = 
We calculate only tunings

'''