'''
These functions will align graph captured to brain atlas.
User need to provide bregma and lambda location.

'''
#%%
import cv2
import numpy as np
import time
from Brain_Atlas.Atlas_Mask import Mask_Generator
import matplotlib.pyplot as plt

#%%
# Define the points A, B, C, D, E
import cv2
import numpy as np

# Load the image
img = np.zeros(shape = (256,256))
img[100:110,:]=1

# Define the special points (A, B, C, D, E)
points = np.array([
    [50, 100],  # Point A
    [70, 110],  # Point B
    [90, 120],  # Point C
    [110, 130], # Point D
    [130, 140]  # Point E
], dtype=np.float32)

# Step 1: Fit a line to the points (optional, if needed for rotation)
x = points[:, 0]
y = points[:, 1]
fit = np.polyfit(x, y, 1)  # Linear fit
m, b = fit  # slope and intercept

# Step 2: Calculate the angle to rotate to vertical
angle = np.arctan(m) * 180 / np.pi  # convert to degrees
rotation_angle = -angle  # we want to rotate counter-clockwise

# Step 3: Rotate the image around the midpoint of A and B
center = np.mean(points[:2], axis=0)  # midpoint between A and B
M = cv2.getRotationMatrix2D((center[0], center[1]), rotation_angle, 1.0)
rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

# Step 4: Rescale the distance between points A and B to 105 pixels
A_rotated = np.dot(M[:, :2], points[0]) + M[:, 2]
B_rotated = np.dot(M[:, :2], points[1]) + M[:, 2]
current_distance = np.linalg.norm(B_rotated - A_rotated)

# Scale factor
desired_distance = 105
scale_factor = desired_distance / current_distance

# Rescale the image
rescaled_img = cv2.resize(rotated_img, None, fx=scale_factor, fy=scale_factor)

# Display the results
cv2.imshow('Original Image', img)
cv2.imshow('Rotated Image', rotated_img)
cv2.imshow('Rescaled Image', rescaled_img)

# Wait for a key press and close windows
cv2.waitKey(150)
cv2.destroyAllWindows()

# New coordinates of point A after transformations
new_A = A_rotated * scale_factor
print(f"New coordinates of Point A: {new_A}")

#%%
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
#%%
# %matplotlib qt
import matplotlib.pyplot as plt

# Create a sample plot
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]
plt.plot(x, y, 'o-')
plt.title('Interactive Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show the plot
plt.show()