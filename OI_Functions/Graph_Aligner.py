'''
These functions provide graph align tools for figure processing.

Motion correction is done.

'''

#%%
'''
First part, boulder detection method. These functions are used for red map boulder detection.
'''
import numpy as np
from scipy.signal import correlate,convolve
from scipy.ndimage import shift
import cv2

def detect_boulders_canny(gray, canny_low=3, canny_high=50):

    # Reduce noise with Gaussian blur
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred = gray
    # Detect edges using Canny
    edges = cv2.Canny(blurred, canny_low, canny_high)

    return edges.astype('f8')

def detect_boulders_sobel(gray,ksize=3):
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    # Step 2: Compute gradient magnitude
    gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_mag = np.uint8(255 * gradient_mag / np.max(gradient_mag))  
    # Normalize to 0-255
    # Step 3: Threshold to retain strong edges
    # _, thresh = cv2.threshold(gradient_mag, thres, 255, cv2.THRESH_BINARY)
    return gradient_mag.astype('f8')


#%%
'''
Part 2 are boulder detection core. 
Designed for uint 16 data, be aware!
Use sobel function. 
'''



def motion_correction(target, template,motion_lim = 5,ksize = 5):

    ## Preprocess for boulder tetection.
    # template_mean_sub = template - np.mean(template)
    # target_mean_sub = target - np.mean(target)
    target_u1 = (target/256).astype('u1')
    template_u1 = (template/256).astype('u1')
    template_mean_sub = detect_boulders_sobel(template_u1,ksize)
    target_mean_sub = detect_boulders_sobel(target_u1,ksize)
    # plt.imshow(target_mean_sub)

    # Compute cross-correlation 
    corr = correlate(template_mean_sub, target_mean_sub, mode='same')
    # corr = convolve(template_mean_sub, target_mean_sub, mode='same')
    
    # Determine the center of the correlation matrix
    center_y, center_x = np.array(corr.shape) // 2
    
    # Extract a 11x11 region around the center (5 pixels in each direction)
    corr_region = corr[center_y - motion_lim : center_y + motion_lim+1, center_x - motion_lim : center_x + motion_lim+1]
    
    # Find the peak within the region
    peak_y, peak_x = np.unravel_index(np.argmax(corr_region), corr_region.shape)
    
    # Calculate the shift from the target to the template
    dx = peak_x - motion_lim  # x shift (columns)
    dy = peak_y - motion_lim  # y shift (rows)
    
    # Ensure shifts are within the ±5 pixel limit (handles edge cases)
    dx = np.clip(dx, -motion_lim, motion_lim)
    dy = np.clip(dy, -motion_lim, motion_lim)
    
    # Shift the target image by (-dx, -dy) to align with the template
    aligned = shift(target, (dy, dx), order=0, mode='constant', cval=0.0)
    
    return dx, dy, aligned


