'''
Here we try to load brain atlas and use it as roi finder.
'''
#%%

import seaborn as sns
import nrrd
import OI_Functions.Common_Functions as cf
import numpy as np
import matplotlib.pyplot as plt

datapath = r'D:\_Allen_Atlas'
savepath = r'D:\_Codes\Lulab_OI_Infrastructure\Brain_Atlas'

top,header_t = nrrd.read(cf.join(wp,'top.nrrd'))
avr,header_avr = nrrd.read(cf.join(wp,'average_template_10.nrrd'))

#%%
import ccf_streamlines.projection as ccfproj

mappath = r'D:\_Codes\Lulab_OI_Infrastructure\Brain_Atlas\Allen'
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

#%% try to show projection to 2D view data.
top_projection_max = proj_top.project_volume(avr)

plt.imshow(
    top_projection_max.T, # transpose so that the rostral/caudal direction is up/down
    interpolation='none',
    cmap='Greys_r',
)
# for k, boundary_coords in bf_left_boundaries.items():
#     plt.plot(*boundary_coords.T, c="white", lw=0.5)
# for k, boundary_coords in bf_right_boundaries.items():
#     plt.plot(*boundary_coords.T, c="white", lw=0.5)

#%% Plot boundaries.
proj_bf = ccfproj.Isocortex2dProjector(
    # Specify our view lookup file
    cf.join(mappath,"top.h5"),
    # Specify our streamline file
    cf.join(mappath,"surface_paths_10_v3.h5"),
    # Specify that we want to project both hemispheres
    hemisphere="both",
    # The butterfly view doesn't contain space for the right hemisphere,
    # but the projector knows where to put the right hemisphere data so
    # the two hemispheres are adjacent if we specify that we're using the
    # butterfly flatmap
    view_space_for_other_hemisphere=False,
)
#%% get boundaries from top view
top_boundary_finder = ccfproj.BoundaryFinder(
    projected_atlas_file=cf.join(mappath,"top.nrrd"),
    labels_file=cf.join(mappath,"labelDescription_ITKSNAPColor.txt"),
)
top_left_boundaries = top_boundary_finder.region_boundaries()
top_right_boundaries = top_boundary_finder.region_boundaries(
    # we want the right hemisphere boundaries, but located in the right place
    # to plot both hemispheres at the same time
    hemisphere='right',
    # view_space_for_other_hemisphere='top'
)
# top_left_boundaries

plt.imshow(
    top_projection_max.T,
    interpolation='none',
    cmap='Greys_r',
)

for k, boundary_coords in top_left_boundaries.items():
    plt.plot(*boundary_coords.T, c="white", lw=0.5)
for k, boundary_coords in top_right_boundaries.items():
    plt.plot(*boundary_coords.T, c="white", lw=0.5)
#%% if we show each brain area seperately

bk = np.zeros(shape = (1320,1140),dtype='f8')
plt.imshow(
    top_projection_max.T,
    interpolation='none',
    cmap='Greys_r',
)
# plt.imshow(bk,cmap='Greys_r')


lv1 = top_left_boundaries['SSp-m']
plt.plot(*lv1.T, c="r", lw=0.5)

# rv1 = top_right_boundaries['VISp']
# plt.plot(*rv1.T, c="b", lw=0.5)
# plt.scatter(x = rv1[:,0],y = rv1[:,1], c="b", s = 1) # scatter method

for k, boundary_coords in top_right_boundaries.items():
    plt.plot(*boundary_coords.T, c="blue", lw=0.5)

#%% can we use this to get brain area mask?

from matplotlib.path import Path


# Create a path from the boundary points
boundary_path = Path(top_left_boundaries['SSp-m'])

# Create a grid of points (x, y) corresponding to the graph
height, width = 1320, 1140
y_indices, x_indices = np.indices((height, width))

# Create an array of points
points = np.vstack((x_indices.ravel(), y_indices.ravel())).T

# Check which points are inside the boundary
inside_mask = boundary_path.contains_points(points)

# Reshape the mask to the shape of the graph
mask = inside_mask.reshape((height, width))

# Optionally, visualize the mask
# plt.imshow(top_projection_max.T,cmap='Greys_r')
# plt.imshow(top_projection_max.T*mask, cmap='gray')
# plt.title('Mask of Pixels Inside Boundary')
# plt.show()

#%% If you want to add text on the graph given.
y_coords, x_coords = np.where(mask)
center_y = int(np.mean(y_coords))
center_x = int(np.mean(x_coords))

# Create a plot to visualize the mask and add text
plt.imshow(mask, cmap='gray')
plt.text(center_x, center_y, 'Center', color='red', fontsize=12, ha='center', va='center')
plt.title('Mask of Pixels Inside Boundary with Text at Center')
plt.show()

#%%
global_mask = top_projection_max.T>0

non_zero_indices = np.argwhere(global_mask == 1)
min_y, min_x = non_zero_indices.min(axis=0)
max_y, max_x = non_zero_indices.max(axis=0)

len(top_left_boundaries.keys())