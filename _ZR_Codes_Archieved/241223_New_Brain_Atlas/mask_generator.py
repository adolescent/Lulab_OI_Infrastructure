'''
This project will generate brain area mask with bregma location.
From Allen atlas get locations.
1 pix is 10um for allen dataset.
'''
#%%

import seaborn as sns
import nrrd
import OI_Functions.Common_Functions as cf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ccf_streamlines.projection as ccfproj # allen projector
from tqdm import tqdm
import copy


datapath = r'D:\_Allen_Atlas'
savepath = r'D:\_Codes\Lulab_OI_Infrastructure\Brain_Atlas'

top_mask,header_t = nrrd.read(cf.join(datapath,'top.nrrd'))
avr,header_avr = nrrd.read(cf.join(datapath,'average_template_10.nrrd'))

mappath = r'D:\_Allen_Atlas'

#%% 
'''
Step 1, get top view projection info.
For this map, 1 pix is 10um.
'''
proj_top = ccfproj.Isocortex2dProjector(
    # Specify our view lookup file
    cf.join(mappath,"top.h5"),
    # Specify our streamline file
    cf.join(mappath,"surface_paths_10_v3.h5"),
    # Specify that we want to project both hemispheres
    hemisphere="both",
    # The top view contains space for the right hemisphere, but is empty.
    # Therefore, we tell the projector to put both hemispheres side-by-side
    view_space_for_other_hemisphere=True,
)

top_projection_max = proj_top.project_volume(avr).T

plt.imshow(top_projection_max,cmap='Greys_r')
cf.Save_Variable(savepath,'Raw_Brain_Top',top_projection_max,'.pkl')
#%% calculate boulder of masks. This is used for bregma calculation.
# bregma is 3.56 mm back of the most front cortex coords.
global_mask = top_projection_max>0

non_zero_indices = np.argwhere(global_mask == 1)
min_y, min_x = non_zero_indices.min(axis=0)
max_y, max_x = non_zero_indices.max(axis=0)

height,width = global_mask.shape
bregma_loc = (min_y+356,int(width/2))

print(f'Left:{min_x},Right:{max_x}')
print(f'Top:{min_y},Bottom:{max_y}')
print(f'Bregma Loc:{bregma_loc}')

cf.Save_Variable(savepath,'Bregma',bregma_loc)
#%%
'''
Step 2, get counter and mask location.
We need to get visible locations first.
'''

top_boundary_finder = ccfproj.BoundaryFinder(projected_atlas_file=cf.join(mappath,"top.nrrd"),labels_file=cf.join(mappath,"labelDescription_ITKSNAPColor.txt"))

boundaries_l = top_boundary_finder.region_boundaries()
boundaries_r = top_boundary_finder.region_boundaries(
    # we want the right hemisphere boundaries, but located in the right place
    # to plot both hemispheres at the same time
    hemisphere='right',
    # view_space_for_other_hemisphere='top'
)

for k, boundary_coords in boundaries_l.items():
    plt.plot(*boundary_coords.T, c='b', lw=0.5)
for k, boundary_coords in boundaries_r.items():
    plt.plot(*boundary_coords.T, c='r', lw=0.5)

# add bregma on graph
from matplotlib.patches import Circle
circle = Circle((bregma_loc[1],bregma_loc[0]),10, color='black',fill=True, zorder=5)
plt.gca().add_patch(circle)

# show graph generated 
plt.imshow(top_projection_max,interpolation='none',cmap='Greys_r')
plt.savefig(cf.join(savepath,'Boundaries.png'),dpi = 300)

cf.Save_Variable(savepath,'Boundaries',(boundaries_l,boundaries_r))

#%% generate all brain area mask
from matplotlib.path import Path
def Boulder2Mask(boundaries_dict,mask_name):
    boundary_path = Path(boundaries_dict[mask_name])
    height, width = 1320, 1140
    y_indices, x_indices = np.indices((height, width))
    points = np.vstack((x_indices.ravel(), y_indices.ravel())).T
    inside_mask = boundary_path.contains_points(points)
    mask = inside_mask.reshape((height, width))
    # then calculate center of mask
    y_coords, x_coords = np.where(mask)
    center_y = int(np.mean(y_coords))
    center_x = int(np.mean(x_coords))
    return mask,(center_y,center_x)

#%% get all visible area masks

area_lists = list(boundaries_l.keys())

min_area = 1000 # there are 6 very small areas that need to be ignored.
mask_lists = pd.DataFrame(columns = ['ID','Area','LR','Mask','Mask_Cen'])
# id_area_lists.loc[len(id_area_lists)] = [0,'Outside','Both']

for i,c_name in tqdm(enumerate(area_lists)):
    c_mask,c_cen = Boulder2Mask(boundaries_l,c_name)
    if c_mask.sum()>min_area:
        mask_lists.loc[len(mask_lists)] = [len(mask_lists)+1,area_lists[i],'L',c_mask,c_cen]
    else:
        print(f'Area {c_name} is too small, we ignore it.')


for i,c_name in tqdm(enumerate(area_lists)):
    c_mask,c_cen = Boulder2Mask(boundaries_r,c_name)
    if c_mask.sum()>min_area:
        mask_lists.loc[len(mask_lists)] = [len(mask_lists)+1,area_lists[i],'R',c_mask,c_cen]
    else:
        print(f'Area {c_name} is too small, we ignore it.')

cf.Save_Variable(savepath,'Brain_Area_Masks',mask_lists)
#%% an easy test of all mask, it's okay. I found several small mask need to be cutted 
for i in tqdm(range(len(mask_lists))):
    c_mask = mask_lists.loc[i,'Mask']
    plt.imshow(c_mask,interpolation='none',cmap='Greys_r')
    plt.savefig(cf.join(savepath,f'{i}.png'))

#%% get stacked mask map
area_map = np.zeros(shape = (1320,1140))
labels = []
for i in range(31):
    c_area = mask_lists.iloc[i,:]
    c_area_r = mask_lists.iloc[i+31,:]
    c_index = c_area['ID']
    c_mask = c_area['Mask']+c_area_r['Mask']
    area_map += c_mask*c_index

    c_name = c_area['Area']
    labels.append(c_name)

# %%plot annotated graphs on graph.
import matplotlib.colors as mcolors
cmap = plt.cm.get_cmap('jet', 32)

img = plt.imshow(area_map,interpolation='none',cmap='jet')
cbar = plt.colorbar(img, ticks=np.arange(1, 32))
cbar.ax.set_yticklabels(labels)  
plt.savefig(cf.join(savepath,'Brain_Areas_A.png'),dpi = 300)

#%% get lined map with brain area annotated

# plot boulder and raw graph
for k, boundary_coords in boundaries_l.items():
    plt.plot(*boundary_coords.T, c='b', lw=0.5)
for k, boundary_coords in boundaries_r.items():
    plt.plot(*boundary_coords.T, c='r', lw=0.5)
# add bregma on graph
from matplotlib.patches import Circle
circle = Circle((bregma_loc[1],bregma_loc[0]),10, color='black',fill=True, zorder=5)
plt.gca().add_patch(circle)

# plot names on graph.
for i in range(len(mask_lists)):
    c_area = mask_lists.iloc[i,:]
    c_name = c_area['Area']
    cen_y,cen_x = c_area['Mask_Cen']
    if c_area['LR'] == 'L':
        plt.text(cen_x, cen_y,c_name, color='b', fontsize=3, ha='center', va='center')
    else:
        plt.text(cen_x, cen_y,c_name, color='r', fontsize=3, ha='center', va='center')

# show graph generated 
plt.imshow(top_projection_max,interpolation='none',cmap='Greys_r')
plt.savefig(cf.join(savepath,'Brain_Areas_B.png'),dpi = 600)
#%% get full-scale id conditioned map.

id_map = np.zeros(shape = (1320,1140))
for i in range(len(mask_lists)):
    c_area = mask_lists.iloc[i,:]
    c_id = c_area['ID']
    c_mask = c_area['Mask']
    id_map += c_mask*c_id
np.save(cf.join(savepath,'Raw_Area_ID'),id_map)

#%% get 22 and 44 binned mask,mask center and id conditioned map.
# id map first.
bin2_idmap = id_map.reshape(660, 2, 570, 2).mean(axis=(1, 3)).astype('i4')
bin4_idmap = id_map.reshape(330, 4, 285, 4).mean(axis=(1, 3)).astype('i4')

np.save(cf.join(savepath,'Raw_Area_ID_bin2'),bin2_idmap)
np.save(cf.join(savepath,'Raw_Area_ID_bin4'),bin4_idmap)

mask_lists_bin2 = copy.deepcopy(mask_lists).drop('Mask',axis=1)
mask_lists_bin4 = copy.deepcopy(mask_lists).drop('Mask',axis=1)




mask_lists_bin2['Mask'] = [np.zeros(shape = (660,570),dtype='bool')]*62
mask_lists_bin4['Mask'] = [np.zeros(shape = (330,285),dtype='bool')]*62


for i in tqdm(range(len(mask_lists))):

    c_mask = mask_lists.loc[i,'Mask']
    c_cen = mask_lists.loc[i,'Mask_Cen']

    mask_lists_bin2['Mask'].loc[i] = c_mask.reshape(660, 2, 570, 2).mean(axis=(1, 3))>0
    mask_lists_bin4['Mask'].loc[i] = c_mask.reshape(330, 4, 285, 4).mean(axis=(1, 3))>0

    mask_lists_bin2['Mask_Cen'].loc[i] = (int(c_cen[0]/2),int(c_cen[1]/2))
    mask_lists_bin4['Mask_Cen'].loc[i] = (int(c_cen[0]/4),int(c_cen[1]/4))


cf.Save_Variable(savepath,'Brain_Area_Masks_bin2',mask_lists_bin2)
cf.Save_Variable(savepath,'Brain_Area_Masks_bin4',mask_lists_bin4)
# cf.Save_Variable(savepath,'Brain_Area_Masks',mask_lists)
    
#%%
from Brain_Atlas.Atlas_Mask import Mask_Generator