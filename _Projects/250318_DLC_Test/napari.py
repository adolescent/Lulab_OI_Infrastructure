'''
Try to fix the opengl bug of napari. 

IT SUCKS
'''
#%%
from skimage import data
import napari
# import vispy

# viewer = napari.view_image(data.cells3d(), channel_axis=1, ndisplay=3)
# napari.run()  # start the "event loop" and show the viewer

# print(vispy.sys_info())
import vispy
vispy.use("jupyter_rfb")
print(vispy.sys_info())
