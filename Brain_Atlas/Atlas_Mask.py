

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

class Mask_Generator(object):

    name = 'Mask of visable parts'

    def __init__(self,bin=4) -> None:# bin is the only require od 

        breg = cf.Load_Variable('Bregma.pkl') # breg in seq y,x
        # self.breg = 
        pass

    def Pix_Label(pix):
        pass

    def Area_Mask(area,LR = 'L'):
        pass

    
#%%
import cv2
import numpy as np
import time

# Create a 512x512 black image
image = np.zeros((512, 512, 3), dtype=np.uint8)

# Variable to track if a point has been selected
point_selected = False
selected_point = None  # Variable to store the selected point coordinates

# Function to capture mouse events
def get_coordinates(event, x, y, flags, param):
    global point_selected, selected_point
    if event == cv2.EVENT_LBUTTONDOWN and not point_selected:
        selected_point = (x, y)  # Store the coordinates in the variable
        print(f"Selected point: {selected_point}")  # Print the coordinates
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Draw a circle on the selected point
        cv2.imshow("Graph", image)  # Update the display
        point_selected = True  # Mark that a point has been selected

# Display the image in a window
cv2.imshow("Graph", image)
cv2.setMouseCallback("Graph", get_coordinates)

# Wait until a point is selected
while not point_selected:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Allow quitting with 'q'

# Suspend for 5 seconds
# if point_selected:
#     time.sleep(3)
# Close the window
cv2.destroyAllWindows()

cv2.imshow("Graph", image)
cv2.waitKey(1)


# Checkpoint
user_input = input("Do you want to continue? (Y/N): ").strip().upper()
if user_input != 'Y':
    cv2.destroyAllWindows()
    raise ValueError("Process terminated by user.")
else:
    print("Continuing with the process...")
    cv2.destroyAllWindows()
    print(f"Using selected point coordinates: {selected_point}")